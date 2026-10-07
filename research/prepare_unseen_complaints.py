"""Create a held-out dynamic-label diagnostic without reusing training or main-test text."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "complaints"
DEST = SOURCE / "unseen"


NEW_LEAVES = [
    ("L28", "A1", "A1B1", "재활용품 분리배출", "종이·플라스틱·유리 등 재활용품의 분리배출 방법과 수거 안내", "new_leaf"),
    ("L29", "A2", "A2B2", "도로공사 안전관리", "도로 공사 구간의 안전펜스·표지·통행 안전 조치", "new_leaf"),
    ("L30", "A3", "A3B2", "복지 상담 예약", "복지 담당자와의 방문·전화 상담 일정 예약과 변경", "new_leaf"),
    ("L31", "A1", "A1B4", "유기동물 구조", "도로와 주택가의 다친 유기동물 구조와 보호 요청", "new_middle"),
    ("L32", "A2", "A2B4", "자전거도로 장애물", "자전거도로를 막는 적치물·차량의 이동과 통행 확보", "new_middle"),
    ("L33", "A3", "A3B4", "지방세 납부", "재산세·자동차세 등 지방세의 납부 방법과 내역 확인", "new_middle"),
    ("L34", "A4", "A4B1", "건축물 붕괴 위험", "균열·기울어짐 등 건축물의 붕괴 위험 점검과 안전 조치", "new_major"),
    ("L35", "A5", "A5B1", "직업훈련 신청", "구직자를 위한 직업교육 과정의 자격·신청·일정 안내", "new_major"),
    ("L36", "A6", "A6B1", "도서 대출 문제", "공공도서관 도서 대출·반납·예약 상태의 확인과 처리", "new_major"),
]


CASES = {
    "L28": [
        ("explicit", "인터넷 상담", "투명 페트병과 일반 플라스틱을 어떻게 나눠 버리는지 수거 기준을 알려 달라고 함."),
        ("implicit", "콜센터", "내용물을 씻은 유리병과 종이팩을 같은 봉투에 넣어도 되는지 문의함."),
        ("mixed_context", "인터넷 상담", "일반 쓰레기는 정상 수거됐고 남은 페트병 묶음의 올바른 배출 요일과 방법만 확인하고 싶어 함."),
    ],
    "L29": [
        ("explicit", "콜센터", "도로 굴착 구간에 안전펜스와 야간 경고등이 없어 보행자 사고가 우려된다고 신고함."),
        ("implicit", "인터넷 상담", "아스팔트 보수 자체보다 작업장 주변 안내 표지와 우회 통로가 없어 위험하다는 요청임."),
        ("mixed_context", "콜센터", "공사 소음은 끝났지만 차로에 남은 작업 장비와 쓰러진 안전콘을 정리해 달라고 함."),
    ],
    "L30": [
        ("explicit", "인터넷 상담", "복지 지원 내용을 담당자와 상담할 방문 시간을 다음 주로 예약하고 싶어 함."),
        ("implicit", "콜센터", "자격이나 서류 답변보다 담당 복지사와 통화 가능한 시간을 먼저 잡아 달라고 함."),
        ("mixed_context", "인터넷 상담", "지원금 입금은 확인됐고 기존 상담 약속만 다른 날짜로 변경해 달라는 요청임."),
    ],
    "L31": [
        ("explicit", "콜센터", "차도 가장자리에 다친 유기견이 움직이지 못하고 있어 구조와 보호를 요청함."),
        ("implicit", "인터넷 상담", "목줄 없는 어린 개가 며칠째 공원을 돌며 다리를 절고 있어 안전하게 데려가 달라고 함."),
        ("mixed_context", "콜센터", "공원 쓰레기 문제도 있지만 지금은 벤치 아래 쓰러진 고양이의 긴급 구조가 우선이라고 함."),
    ],
    "L32": [
        ("explicit", "인터넷 상담", "자전거 전용도로에 적재물이 쌓여 통행할 수 없으니 장애물을 치워 달라고 함."),
        ("implicit", "콜센터", "두 바퀴로 지나가는 구간을 공사 자재가 모두 막아 차도로 내려가야 한다고 설명함."),
        ("mixed_context", "인터넷 상담", "불법주차 단속 문의가 아니라 자전거 길 한가운데 세워진 공유 킥보드를 이동해 달라는 요청임."),
    ],
    "L33": [
        ("explicit", "콜센터", "자동차세 고지서의 납부 기한과 온라인 납부 방법을 알려 달라고 함."),
        ("implicit", "인터넷 상담", "구청에서 부과한 재산 관련 세금을 어느 계좌로 보내야 하는지 문의함."),
        ("mixed_context", "콜센터", "복지 자격 상담은 끝났고 이번 전화에서는 체납된 지방세 내역만 확인하고 싶어 함."),
    ],
    "L34": [
        ("explicit", "인터넷 상담", "오래된 상가 외벽에 큰 균열이 생기고 건물이 기울어 붕괴 위험 점검을 요청함."),
        ("implicit", "콜센터", "비가 샌다는 수준이 아니라 벽이 벌어지고 천장에서 콘크리트 조각이 떨어져 긴급 안전 확인이 필요함."),
        ("mixed_context", "인터넷 상담", "도로 파손 신고가 아니라 도로 옆 빈 건물이 한쪽으로 기울어 보행자를 덮칠까 걱정된다는 민원임."),
    ],
    "L35": [
        ("explicit", "콜센터", "실직 후 참여할 수 있는 용접 직업훈련 과정의 신청 자격과 일정을 문의함."),
        ("implicit", "인터넷 상담", "새 일자리를 구하기 전에 지자체가 지원하는 기술 교육을 어디서 신청하는지 알고 싶어 함."),
        ("mixed_context", "콜센터", "생계 지원금 상담은 마쳤고 현재 요청은 취업에 필요한 교육 과정 등록 방법 안내임."),
    ],
    "L36": [
        ("explicit", "인터넷 상담", "도서관에서 반납한 책이 계속 대출 중으로 표시되어 상태 확인을 요청함."),
        ("implicit", "콜센터", "예약한 책이 도착했다는 알림을 받았는데 창구에서는 예약 기록이 없다고 하여 확인을 원함."),
        ("mixed_context", "인터넷 상담", "도서관 운영시간 불만이 아니라 반납함에 넣은 책에 연체료가 붙은 문제를 처리해 달라고 함."),
    ],
}


def main():
    taxonomy = json.loads((SOURCE / "taxonomy.json").read_text("utf-8"))
    taxonomy["notes"] += " L28-L36 are inference-only labels absent from all training and validation rows."
    majors = {major["id"]: major for major in taxonomy["majors"]}
    new_middle_specs = [
        ("A1", "A1B4", "동물보호", ["L31"]),
        ("A2", "A2B4", "자전거·보행", ["L32"]),
        ("A3", "A3B4", "지방세", ["L33"]),
    ]
    for major_id, middle_id, name, children in new_middle_specs:
        majors[major_id]["children"].append({"id": middle_id, "name": name, "children": children})
    for major_id, name, middle_id, middle_name, leaf_id in [
        ("A4", "주택·건축", "A4B1", "건축물 안전", "L34"),
        ("A5", "경제·일자리", "A5B1", "취업지원", "L35"),
        ("A6", "교육·문화", "A6B1", "도서관 서비스", "L36"),
    ]:
        taxonomy["majors"].append({"id": major_id, "name": name, "children": [{"id": middle_id, "name": middle_name, "children": [leaf_id]}]})
    for leaf_id, major_id, middle_id, name, description, novelty in NEW_LEAVES:
        if novelty == "new_leaf":
            middle = next(node for node in majors[major_id]["children"] if node["id"] == middle_id)
            middle["children"].append(leaf_id)
        major_name = next(major["name"] for major in taxonomy["majors"] if major["id"] == major_id)
        middle_name = next(node["name"] for major in taxonomy["majors"] for node in major["children"] if node["id"] == middle_id)
        taxonomy["leaves"].append({
            "id": leaf_id,
            "major_id": major_id,
            "middle_id": middle_id,
            "path": [major_name, middle_name, name],
            "description": description,
            "novelty": novelty,
        })
    cases = []
    for leaf_id, rows in CASES.items():
        novelty = next(item[5] for item in NEW_LEAVES if item[0] == leaf_id)
        for index, (slice_name, channel, summary) in enumerate(rows, 1):
            cases.append({
                "id": f"{leaf_id}-{slice_name}",
                "channel": channel,
                "slice": slice_name,
                "novelty": novelty,
                "summary": summary,
                "expected_leaf": leaf_id,
                "acceptable_leaves": [leaf_id],
                "needs_clarification": False,
            })
    training_text = "\n".join((SOURCE / "training" / name).read_text("utf-8") for name in ["train.jsonl", "validation.jsonl"])
    main_test = (SOURCE / "cases.json").read_text("utf-8")
    assert all(case["summary"] not in training_text and case["summary"] not in main_test for case in cases)
    assert len(cases) == 27 and len({case["summary"] for case in cases}) == 27
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    (DEST / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    manifest = {
        "purpose": "Inference-only dynamic-label diagnostic",
        "counts": {"existing_leaves": 27, "new_leaves": 9, "cases": 27, "per_new_label": 3},
        "novelty": {"new_leaf": 3, "new_middle": 3, "new_major": 3},
        "training_use": False,
        "exact_text_overlap_with_training_validation_or_main_test": False,
        "limitations": ["Author-written synthetic data", "No independent human review", "Three cases per new label"],
        "sha256": {name: hashlib.sha256((DEST / name).read_bytes()).hexdigest() for name in ["taxonomy.json", "cases.json"]},
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
