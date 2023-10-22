def get_crit_info_lang(opt):
    from .crit_lang import LanguageGeneration
    object_func = [LanguageGeneration(opt)]
    name = ['Lang Loss']
    scale = [opt.get('language_generation_scale', 1.0)]
    return object_func, name, scale


def get_crit_info_length(opt):
    from .crit_length import KLDivLoss
    objects = [KLDivLoss(opt)]
    names = ['Length Loss']
    scales = [opt.get('length_prediction_scale', 1.0)]
    return objects, names, scales


def get_crit_info_attribute(opt):
    from .crit_attribute import NoisyOrMIL, NoisyOrMILWithEmbs

    # embs_name, prefix of fieldsnames, loss_name, loss_scale
    mappings = {
        'P': "input_embs_exclude_bos",
        'I': "input_embs",
        'C': "context",
        'H': "hidden_states",
        'T': "text_context",
        'S': "sentence_embs",
        'A': "attr_embs",
    }

    objects = []
    names = []
    scales = opt.get('attribute_prediction_scales', 1.0)
    flags = opt['attribute_prediction_flags']

    if not isinstance(scales, list):
        scales = [scales]
    elif len(scales) == 1:
        scales = scales * len(flags)
    else:
        assert len(scales) == len(flags), f'#scales {len(scales)} vs. #flags {len(flags)}'

    for flag in flags:
        names.append(f'{flag}-Attr')

        if flag == 'V':
            objects.append(NoisyOrMIL(opt))
        else:
            assert flag in mappings, f'- We can not find {flag} in mappings: {mappings.keys()}'
            objects.append(NoisyOrMILWithEmbs(opt, keys=mappings[flag], flag=flag, prefix=f'{flag}-'))

    return objects, names, scales
