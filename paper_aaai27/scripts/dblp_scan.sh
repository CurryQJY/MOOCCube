#!/usr/bin/env bash
# Structured DBLP scan for CKG-RL related work (top venues, 2025-2026)
set -u

OUT="D:/DeskTop/MOOCCube/paper_aaai27/scripts/dblp_hits.tsv"
: > "$OUT"

# Top venues we care about (regex, case-insensitive match on venue field)
VEN_RE='AAAI|SIGIR|KDD|NeurIPS|NIPS|ICML|ICLR|WWW|The Web Conference|WSDM|IJCAI|CIKM|RecSys|ACL|EMNLP|NAACL'

queries=(
  "cold-start+item+recommendation"
  "cold+start+recommendation"
  "course+recommendation"
  "MOOC+recommendation"
  "learning+path+recommendation"
  "educational+recommendation"
  "prerequisite+recommendation"
  "knowledge+graph+cold+start"
  "user+simulation+recommendation"
  "user+simulator+recommendation"
  "reinforcement+learning+recommendation"
  "offline+reinforcement+learning+recommendation"
  "reward+shaping+recommendation"
  "content+based+cold+start"
  "meta+learning+cold+start"
  "new+item+recommendation"
  "item+exposure+fairness+recommendation"
  "popularity+bias+cold+start"
  "sequential+recommendation+cold+start"
  "LLM+cold+start+recommendation"
  "retrieval+augmented+recommendation"
  "graph+neural+network+cold+start"
  "curriculum+reinforcement+learning"
  "exploration+recommendation"
)

for q in "${queries[@]}"; do
  ok=0
  for attempt in 1 2 3 4; do
    resp=$(curl -s --max-time 30 \
      "https://dblp.org/search/publ/api?q=${q}&h=200&format=json" 2>/dev/null)
    if [ -n "$resp" ] && printf '%s' "$resp" | head -c1 | grep -q '{'; then
      printf '%s' "$resp" > "D:/DeskTop/MOOCCube/paper_aaai27/scripts/.dblp_tmp.json"
      python - "$q" "$VEN_RE" <<'PY'
import json,sys,re
q=sys.argv[1]; ven_re=re.compile(sys.argv[2], re.I)
try:
    d=json.load(open("D:/DeskTop/MOOCCube/paper_aaai27/scripts/.dblp_tmp.json",encoding="utf-8"))
except Exception:
    sys.exit(0)
hits=d.get("result",{}).get("hits",{}).get("hit",[])
out=open("D:/DeskTop/MOOCCube/paper_aaai27/scripts/dblp_hits.tsv","a",encoding="utf-8")
for h in hits:
    info=h.get("info",{})
    ven=info.get("venue","")
    ven_s=ven if isinstance(ven,str) else " ".join(ven)
    year=info.get("year","")
    title=info.get("title","").rstrip(".")
    typ=info.get("type","")
    ee=info.get("ee","")
    if not ven_re.search(ven_s): continue
    if year not in ("2025","2026"): continue
    out.write(f"{year}\t{ven_s}\t{title}\t{ee}\t{q}\n")
out.close()
PY
      ok=1
      break
    fi
    sleep 3
  done
  [ $ok -eq 1 ] && echo "OK  $q" || echo "ERR $q"
  sleep 2
done

echo "---- unique hits ----"
sort -u -t$'\t' -k3,3 "$OUT" | sort -t$'\t' -k2,2 -k1,1 | wc -l
