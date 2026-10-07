"""Dependency-free helpers for the 3x3x3 complaint taxonomy."""


def question(criteria):
    return {'route': {'type': 'choice', 'instructions':
        '민원 요약에서 현재 가장 우선적으로 해결해 달라는 요청을 분류하세요. 이미 해결된 문제나 배경 설명보다 현재 요청을 우선하세요. 가장 적합한 유형 하나를 고르세요.',
        'criteria': criteria}}


def build_questions(taxonomy):
    leaves = {x['id']: x for x in taxonomy['leaves']}
    majors = {x['id']: x for x in taxonomy['majors']}
    flat = {k: x['path'][2] + ': ' + x['description'] for k, x in leaves.items()}
    top = {k: m['name'] + ': ' + ', '.join(x['name'] for x in m['children']) for k, m in majors.items()}
    middle, bottom = {}, {}
    for major_id, major in majors.items():
        middle[major_id] = {node['id']: node['name'] + ': ' + ', '.join(leaves[k]['path'][2] for k in node['children'])
                            for node in major['children']}
        for node in major['children']:
            bottom[node['id']] = {k: flat[k] for k in node['children']}
    return leaves, majors, {'flat': flat, 'top': top, 'middle': middle, 'bottom': bottom}
