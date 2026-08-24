#!/bin/bash
# 手动拉取 ghcr.io/btankut/sglang-spark-glm47（curl 断点续传方案）
# 用法: pull_layers.sh <test|download|assemble>
set -u
MODE="${1:-test}"
DIR=/tmp/routeb_sglang
mkdir -p "$DIR/layers"

TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:btankut/sglang-spark-glm47:pull" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
echo "$TOKEN" > "$DIR/token.txt"

if [ "$MODE" = "test" ]; then
    DIGEST=$(python3 - <<'EOF'
import json
m = json.load(open('/tmp/routeb_sglang/manifest.json'))
print(m['layers'][12]['digest'])
EOF
)
    echo "digest: $DIGEST"
    timeout 90 curl -sL -H "Authorization: Bearer $TOKEN" -r 0-20971519 \
        -o "$DIR/speedtest.bin" \
        -w "HTTP %{http_code}, speed: %{speed_download} B/s, got %{size_download} bytes in %{time_total}s\n" \
        "https://ghcr.io/v2/btankut/sglang-spark-glm47/blobs/$DIGEST"
    ls -la "$DIR/speedtest.bin"; rm -f "$DIR/speedtest.bin"
    exit 0
fi

if [ "$MODE" = "download" ]; then
    python3 - <<'EOF' > "$DIR/layers.tsv"
import json
m = json.load(open('/tmp/routeb_sglang/manifest.json'))
for i, l in enumerate(m['layers']):
    print(f"{i}\t{l['digest']}\t{l['size']}")
EOF
    TOTAL=$(python3 -c "import json; m=json.load(open('/tmp/routeb_sglang/manifest.json')); print(sum(l['size'] for l in m['layers']))")
    echo "total bytes: $TOTAL"
    # 3 个大层并行，其余串行；每个 curl 用 -C - 断点续传，失败重试 20 次
    while IFS=$'\t' read -r idx digest size; do
        f="$DIR/layers/layer_${idx}_${digest#sha256:}.tar.gz"
        for attempt in $(seq 1 20); do
            have=0
            [ -f "$f" ] && have=$(stat -c%s "$f")
            if [ "$have" -ge "$size" ]; then echo "layer $idx DONE ($have bytes)"; break; fi
            echo "layer $idx attempt $attempt: have $have / $size"
            curl -sL -H "Authorization: Bearer $TOKEN" -C - -o "$f" \
                "https://ghcr.io/v2/btankut/sglang-spark-glm47/blobs/$digest" && true
        done
    done < "$DIR/layers.tsv"
    echo "ALL DOWNLOADS FINISHED"
    ls -la "$DIR/layers/"
fi
