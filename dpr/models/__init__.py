import importlib


def init_hf_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_biencoder_components

    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_reader(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_reader_components

    return get_bert_reader_components(args, **kwargs)


def init_hf_bert_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_tensorizer

    return get_bert_tensorizer(args)


def init_hf_roberta_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_roberta_tensorizer

    return get_roberta_tensorizer(args)


BIENCODER_INITIALIZERS = {
    "hf_bert": init_hf_bert_biencoder,
}

READER_INITIALIZERS = {
    "hf_bert": init_hf_bert_reader,
}

TENSORIZER_INITIALIZERS = {
    "hf_bert": init_hf_bert_tenzorizer,
    "hf_roberta": init_hf_roberta_tenzorizer,
    "pytext_bert": init_hf_bert_tenzorizer,  # using HF's code as of now
    "fairseq_roberta": init_hf_roberta_tenzorizer,  # using HF's code as of now
}


def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError("unsupported model type: {}".format(type))


def init_biencoder_components(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)


def init_reader_components(encoder_type: str, args, **kwargs):
    return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)
