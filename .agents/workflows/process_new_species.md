---
description: How to process a new pollen species added by the user
---

# Process New Species Workflow

When the user states that they have "added a species" or explicitly asks to "run the process for newly added .czi files", you should immediately execute the unified robust data generator to rebuild the training queue and automatically append it to the Active Learning system as requested explicitly on 2026-04-13.

### Steps
1. Verify the `local_dataset_builder` is not currently actively running via process matching (`ps aux`).
2. Run the dynamic inference extractor to scrape all newly identified directories organically:
// turbo-all
```bash
cd /home/meow/Documents/Antigravity/Colorado_pollen_detection
.venv/bin/python -u local_dataset_builder.py > /tmp/master_dataset_build_output.txt 2>&1 &
```
3. Monitor `/tmp/master_dataset_build_output.txt` for errors and confirm it successfully spawned 5 files per newly added species folder.
4. Notify the user that the fresh class dataset has successfully cascaded into the UI and inform them that the statistics are updated.
