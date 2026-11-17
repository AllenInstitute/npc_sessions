#!/bin/bash
# detect + compare for every fetched session; log to compare_all.log
PY=C:/data/t2p_qc/venv_analog/Scripts/python
SESS="DRpilot_744279_20250116 DRpilot_762526_20250320 DRpilot_796012_20250714 DRpilot_807082_20250930 DRpilot_810752_20251010 DRpilot_814669_20251113 DRpilot_813586_20251205 DRpilot_818720_20251219 DRpilot_836729_20260216 DRpilot_840160_20260501 DRpilot_837230_20260422"
: > compare_all.log
for f in $SESS; do
  $PY detect.py $f > /dev/null
  $PY compare.py $f >> compare_all.log 2>&1
done
echo ALLDONE >> compare_all.log
