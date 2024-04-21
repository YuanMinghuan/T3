import re

from rouge import Rouge


def rouge_score(hypotheses, references):
    assert len(hypotheses) == len(references)
    results = {
        'rouge-1': {
            'r': 0.0,
            'p': 0.0,
            'f': 0.0
        },
        'rouge-2': {
            'r': 0.0,
            'p': 0.0,
            'f': 0.0
        },
        'rouge-l': {
            'r': 0.0,
            'p': 0.0,
            'f': 0.0
        }
    }
    rouge_scorer = Rouge()
    for hyp, ref in zip(hypotheses, references):
        if re.fullmatch(pattern='\.*', string=hyp):
            continue
        else:
            result = rouge_scorer.get_scores(hyps=hyp, refs=ref)[0]
            results['rouge-1']['r'] += result['rouge-1']['r']
            results['rouge-1']['p'] += result['rouge-1']['p']
            results['rouge-1']['f'] += result['rouge-1']['f']
            results['rouge-2']['r'] += result['rouge-2']['r']
            results['rouge-2']['p'] += result['rouge-2']['p']
            results['rouge-2']['f'] += result['rouge-2']['f']
            results['rouge-l']['r'] += result['rouge-l']['r']
            results['rouge-l']['p'] += result['rouge-l']['p']
            results['rouge-l']['f'] += result['rouge-l']['f']

    results['rouge-1']['r'] /= len(hypotheses)
    results['rouge-1']['p'] /= len(hypotheses)
    results['rouge-1']['f'] /= len(hypotheses)
    results['rouge-2']['r'] /= len(hypotheses)
    results['rouge-2']['p'] /= len(hypotheses)
    results['rouge-2']['f'] /= len(hypotheses)
    results['rouge-l']['r'] /= len(hypotheses)
    results['rouge-l']['p'] /= len(hypotheses)
    results['rouge-l']['f'] /= len(hypotheses)

    return results
