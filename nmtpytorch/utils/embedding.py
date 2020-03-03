import torch
import numpy as np
import fasttext
from tqdm import tqdm
from sklearn.decomposition import PCA

import logging
logger = logging.getLogger('nmtpytorch')

def load_fasttext(vocab, bin_fname, normalize=True, center_type='none', sub_pca=False, pca_size=None):
    '''
    load embeddings of words in the given vocab.

    Embeddings for special tokens are desined as follow:
    BOS: all zeros
    EOS: trained as part of fastText
    UNK: average of all the embeddings not part of the vocabulary
    PAD: all zeros

    Embeddings are also post-processed with algorithm proposed Mu et al., 2018.
    All-but-the-Top: Simple and Effective Postprocessing for Word Representations
    https://openreview.net/pdf?id=HkuGJ3kCb
    '''
    sub_center = center_type != 'none'

    # bin_fname might be Path object
    bin_fname = str(bin_fname)
    logger.info(f'load fastText embedding from {bin_fname}')

    model = fastText.load_model(bin_fname)

    vocab_size = len(vocab)
    emb_size = model.get_dimension()
    logger.info(f'vocab size: {vocab_size}, embedding size: {emb_size}')

    # index 0 to 3 represents special TOKENS
    embeddings = [model.get_word_vector(k) for k in list(vocab.tokens())[4:]]

    # UNK: average of all the embeddings not part of the vocabulary
    unks = [model.get_word_vector(token) for token in tqdm(model.get_words()) if token not in vocab._map]
    if len(unks) == 0:
        # if no unseen words found in fastText model, use mean vector over all vocabulary as unk vector
        raise Exception('No unseen words found.')
    else:
        unk_vec = np.stack(unks).mean(axis=0)

    # EOS: trained as part of fastText (</s>)
    eos_vec = model.get_word_vector('</s>')

    embeddings = np.append([eos_vec, unk_vec], embeddings, axis=0)

    # centering: substract mean of all embeddings
    if sub_center:
        logger.info('substract center')
        embeddings -= embeddings.mean(axis=0)

    # substract main PCA components
    if sub_pca:
        if pca_size == None:
            pca_size = emb_size // 100 if emb_size >= 100 else 1
        logger.info(f'substract PCA components (n={pca_size})')

        pca = PCA(n_components=pca_size)
        pca.fit(embeddings)

        # use tensor to calculate vecotrs for each PCA components
        emb_pca = pca.transform(embeddings).reshape(-1, pca_size, 1) * \
            pca.components_.reshape(1, pca_size, -1)
        # and accumulate them to get final PCA values to substract
        emb_pca = emb_pca.sum(axis=-2)

        embeddings -= emb_pca

    # PAD, BOS: all zeros
    bos_vec = np.zeros(emb_size)
    pad_vec = np.zeros(emb_size)

    embeddings = torch.from_numpy(np.append([pad_vec, bos_vec], embeddings, axis=0))

    # normalizing
    if normalize:
        logger.info(f'normalize embedding')
        embeddings = torch.functional.F.normalize(embeddings)

    return embeddings.float()


def load_param_from_fasttext(vocab, args):
    bin_fname = args['pretrained_file']
    center_type = args['center_type']
    sub_pca = args['sub_pca']
    n_pca = args['n_pca']

    emb = load_fasttext(vocab=vocab, bin_fname=bin_fname,
        normalize=True, center_type=center_type, sub_pca=sub_pca, pca_size=n_pca).cuda()

    p = torch.nn.Parameter(emb)
    return p
