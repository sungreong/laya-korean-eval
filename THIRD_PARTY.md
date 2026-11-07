# External sources and redistribution

This is an independent evaluation project, not an official LAYA release.

- [LAYA source](https://github.com/NandhaKishorM/laya), commit `8a6e1328cce2460a0e5aa348ad465bb1b5821cd2`: Apache-2.0. Downloaded into ignored `research/laya/`; its LICENSE and notices are retained.
- [LAYA Multilingual weights](https://huggingface.co/convaiinnovations/laya-multilingual), revision `1720e3e3357cfe1e281542e223f8273b0890ca34`: downloaded separately. Consult the model card and its license before redistribution or deployment.
- [NSMC](https://github.com/e9t/nsmc), commit `cc0670e872d4ac27bfe36c87456783004b39ef6c`: downloaded from the original repository. Source review texts and checkpoints are not committed here. Follow the original dataset's terms. Published sentiment predictions contain sample IDs and model outputs, not review text.
- [SKTBrain KoBERT](https://github.com/SKTBrain/KoBERT), commit `fcd729f2f4b37858f333597c0782388ada51eb5f`, and [the Hugging Face checkpoint](https://huggingface.co/skt/kobert-base-v1), revision `359874884642d748079d4dd4ff547f2cdbff67d6`: Apache-2.0. Model files are downloaded into an ignored directory and are not redistributed by this repository.
- [Kakao Kanana-2 V-Decide 3B](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611), revision `1489062def4145ae3a0a9261fbac4f4d88a8c6d1`: Kanana Open License. The 7.33 GB BF16 checkpoint is downloaded into an ignored directory and is not redistributed by this repository. Review Kakao's model card and license before commercial deployment.
- Synthetic department diagnostics and bilingual token examples are included in `research/prepare_eval.py`; they have not received independent human validation.
- `datasets/complaints/` contains 90 newly authored fictional complaint summaries, synthetic training/validation rows, and a custom 27-leaf taxonomy. These are not private customer records or an official municipal taxonomy; they have not received independent human validation.

This repository does not relicense the upstream model, source code, or datasets. No blanket license is assigned to third-party material.
