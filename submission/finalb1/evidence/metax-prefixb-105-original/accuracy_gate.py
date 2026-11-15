"""Validate the complete official Level 3 receipt without executing EvalScope."""
import argparse
import json
import math
from pathlib import Path

import hashlib

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition: raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def check(root):
    reports = list(root.glob('**/reports/minicpm/math_500.json'))
    require(len(reports) == 1, 'Exactly one new Math-500 report required')
    report_path = reports[0]
    run = report_path.parents[2]
    prediction_path = run / 'predictions/minicpm/math_500_Level 3.jsonl'
    review_path = run / 'reviews/minicpm/math_500_Level 3.jsonl'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    summary = report['execution_summary']
    require(report['dataset_name'] == 'math_500' and report['model_name'] == 'minicpm'
            and report['num'] == 105 and summary['incomplete'] is False, 'Unexpected or incomplete accuracy report')
    for counters in (summary, summary['subsets']['Level 3']):
        require(counters['requested'] == counters['succeeded'] == 105 and counters['errored'] == 0,
                'Accuracy execution did not complete all 105 requests')
    metric, = (row for row in report['metrics'] if row['identity']['name'] == 'accuracy')
    require(metric['num'] == 105 and math.isfinite(metric['score']) and 0.95 <= metric['score'] <= 1,
            'Accuracy score below gate or invalid')
    values, questions = [], set()
    pred, review = rows(prediction_path), rows(review_path)
    require(len(pred) == len(review) == 105, 'Incomplete prediction or review rows')
    prediction_by_index = {row['index']: row for row in pred}
    review_by_index = {row['index']: row for row in review}
    require(set(prediction_by_index) == set(review_by_index) == set(range(105)), 'Missing or repeated indices')
    for index in range(105):
        prediction = prediction_by_index[index]
        output = prediction['model_output']
        review_sample = review_by_index[index]['sample_score']
        score = review_sample['score']
        question = prediction['metadata']['question_id']
        require(output['error'] is None and output['choices'] and prediction['model'] == 'minicpm'
                and score['status'] == 'success' and score['value']['accuracy'] in (0, 1), 'Invalid evaluation row')
        require(question == review_sample['sample_metadata']['question_id'] and question not in questions,
                'Incorrect prediction/review pairing')
        questions.add(question)
        values.append(score['value']['accuracy'])
    measured = sum(values) / 105
    require(measured >= 0.95 and abs(measured - metric['score']) <= 0.000051, 'Accuracy does not match raw reviews')
    return dict(passed=True, count=105, correct=int(sum(values)), accuracy=measured,
                sha256={str(p.relative_to(root)): digest(p) for p in (report_path, prediction_path, review_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))


