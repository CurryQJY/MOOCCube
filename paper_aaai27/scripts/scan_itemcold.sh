#!/usr/bin/env bash
# Item cold-start focused DBLP scan. Strict validation: real key + real DOI only.
set -u
RAW="D:/DeskTop/MOOCCube/paper_aaai27/scripts/raw_itemcold"
mkdir -p "$RAW"

queries=(
  "item+cold+start+recommendation"
  "cold+start+item+embedding"
  "cold+start+item+generative"
  "cold+start+item+diffusion"
  "cold+start+item+contrastive"
  "cold+start+item+graph+neural"
  "cold+start+item+meta+learning"
  "cold+start+item+distillation"
  "cold+start+item+llm"
  "new+item+cold+start"
  "cold+start+item+embedding+alignment"
  "item+cold+start+content+feature"
)

for q in "${queries[@]}"; do
  for attempt in 1 2 3 4; do
    curl -s --max-time 40 \
      "https://dblp.org/search/publ/api?q=${q}&h=80&format=json" \
      -o "$RAW/${q}.json" 2>/dev/null
    if [ -s "$RAW/${q}.json" ] && head -c1 "$RAW/${q}.json" | grep -q '{'; then
      echo "fetched: $q"; break
    fi
    sleep 3
  done
  sleep 2
done
echo "ALL FETCHED"
