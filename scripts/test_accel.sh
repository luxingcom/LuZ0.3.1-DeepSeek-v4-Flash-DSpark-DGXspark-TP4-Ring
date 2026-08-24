#!/bin/bash
# 测试 NJU 镜像对 ghcr 仓库的代理能力（带 token 认证）
set -u
DIR=/tmp/routeb_sglang

echo "=== NJU token ==="
TOKEN=$(curl -s "https://ghcr.nju.edu.cn/token?scope=repository:btankut/sglang-spark-glm47:pull" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" 2>/dev/null)
echo "token-len: ${#TOKEN}"

if [ -n "$TOKEN" ]; then
    echo "=== NJU manifest ==="
    timeout 20 curl -s -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json" \
        -o "$DIR/manifest_nju.json" -w "HTTP %{http_code}, size %{size_download}\n" \
        "https://ghcr.nju.edu.cn/v2/btankut/sglang-spark-glm47/manifests/latest"
    head -c 300 "$DIR/manifest_nju.json" 2>/dev/null; echo

    echo "=== NJU blob speed test (20MB range) ==="
    DIGEST="sha256:20ae40effc0b94af636889ae49ab2692aac7e5fa826aaa69f14e5b036d6534f3"
    timeout 60 curl -sL -H "Authorization: Bearer $TOKEN" -r 0-20971519 \
        -o "$DIR/speedtest_nju.bin" \
        -w "HTTP %{http_code}, speed %{speed_download} B/s, got %{size_download} bytes in %{time_total}s\n" \
        "https://ghcr.nju.edu.cn/v2/btankut/sglang-spark-glm47/blobs/$DIGEST"
    ls -la "$DIR/speedtest_nju.bin" 2>/dev/null
    rm -f "$DIR/speedtest_nju.bin"
fi

echo "=== 检查 ghcr 重定向目标 CDN ==="
GTOKEN=$(curl -s "https://ghcr.io/token?scope=repository:btankut/sglang-spark-glm47:pull" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
DIGEST="sha256:20ae40effc0b94af636889ae49ab2692aac7e5fa826aaa69f14e5b036d6534f3"
timeout 20 curl -s -H "Authorization: Bearer $GTOKEN" -o /dev/null -w "redirect-to: %{redirect_url}\n" \
    "https://ghcr.io/v2/btankut/sglang-spark-glm47/blobs/$DIGEST"

echo "=== ghcr CDN 并行 4 路 range 测速 (各 10MB) ==="
for i in 0 1 2 3; do
    START=$((i * 10485760 + 104857600))
    END=$((START + 10485759))
    (timeout 60 curl -sL -H "Authorization: Bearer $GTOKEN" -r $START-$END \
        -o "$DIR/pt_$i.bin" \
        -w "conn $i: HTTP %{http_code}, %{size_download} bytes in %{time_total}s (%{speed_download} B/s)\n" \
        "https://ghcr.io/v2/btankut/sglang-spark-glm47/blobs/$DIGEST" &)
done
wait
rm -f "$DIR"/pt_*.bin
