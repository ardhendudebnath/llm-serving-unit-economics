#!/usr/bin/env bash
# Bring up Prometheus and Grafana against the vLLM deployment in k3s, on WSL2.
#
#   tr -d '\r' < deploy/observability/wsl-up.sh > /root/wsl-up.sh
#   OBS=/path/to/repo/deploy/observability bash /root/wsl-up.sh
#
# Run inside WSL as root, from a CR-stripped copy for the same reason as the
# other scripts here.
#
# docker-compose.yml describes the stack. This runs the same two containers
# with plain podman, because WSL has no compose provider installed. It wires
# the two names prometheus.yml scrapes to the WSL host, so that file is used
# unchanged:
#
#   vllm:8000            kubectl port-forward to the vllm-server Service
#   dcgm-exporter:9400   wsl-gpu-exporter.py, because DCGM cannot run on WSL2
#
# The port-forward also serves the load generator on Windows, through WSL's
# localhost forwarding.
#
# prometheus.yml's kubernetes-pods job needs in-cluster credentials and finds
# no targets from out here. Prometheus logs that and carries on; the vllm job
# above is what scrapes the pod.
set -uo pipefail

OBS=${OBS:-$(cd "$(dirname "$0")" && pwd)}
say() { echo "[$(date -u '+%H:%M:%S')] $*"; }

# Copied off the Windows mount so the containers never read through it.
rm -rf /root/observability && cp -r "$OBS" /root/observability || exit 1

# In a loop, because kubectl port-forward binds to one pod when it starts and
# exits when that pod goes. The first pod here was replaced two minutes after
# going Ready. The WSL VM ran out of memory while this script pulled images
# during the model load, and the kubelet failed to re-admit the pod after it
# recovered. A one-shot forward would have silently pointed at nothing.
#
# On a 15 GB VM, do not pull images while the model is loading.
say "port-forward svc/vllm-server on :8000, restarted whenever the pod changes"
if [ -f /root/port-forward.pid ]; then
    kill -- -"$(cat /root/port-forward.pid)" 2>/dev/null || true
fi
pkill -f '^k3s kubectl port-forward --address 0.0.0.0 svc/vllm-server' || true
nohup setsid bash -c 'while true; do
    k3s kubectl port-forward --address 0.0.0.0 svc/vllm-server 8000:8000
    sleep 2
done' > /root/port-forward.log 2>&1 < /dev/null &
echo $! > /root/port-forward.pid

say "GPU exporter on :9400"
pkill -f '^python3 /root/observability/wsl-gpu-exporter.py' || true
nohup setsid python3 /root/observability/wsl-gpu-exporter.py --port 9400 \
    > /root/gpu-exporter.log 2>&1 < /dev/null &

podman network exists observability || podman network create observability >/dev/null
for c in prometheus grafana; do
    podman rm -f "$c" >/dev/null 2>&1 || true
done

say "prometheus on :9090"
podman run -d --name prometheus --network observability -p 9090:9090 \
    --add-host vllm:host-gateway --add-host dcgm-exporter:host-gateway \
    -v /root/observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro \
    -v /root/observability/rules.yml:/etc/prometheus/rules.yml:ro \
    docker.io/prom/prometheus:v3.1.0 \
    --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time=15d --web.enable-lifecycle >/dev/null

say "grafana on :3000"
podman run -d --name grafana --network observability -p 3000:3000 \
    -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
    -e GF_AUTH_DISABLE_LOGIN_FORM=true -e GF_USERS_DEFAULT_THEME=light \
    -v /root/observability/grafana/provisioning:/etc/grafana/provisioning:ro \
    -v /root/observability/grafana-dashboard.json:/var/lib/grafana/dashboards/serving.json:ro \
    docker.io/grafana/grafana:11.5.1 >/dev/null

sleep 8
say "checks"
curl -s -m 5 -o /dev/null -w 'vllm via port-forward: %{http_code}\n' http://localhost:8000/health
echo "gpu exporter series: $(curl -s -m 5 localhost:9400/metrics | grep -c '^DCGM_')"
curl -s -m 5 -o /dev/null -w 'prometheus ready: %{http_code}\n' http://localhost:9090/-/ready
curl -s -m 5 -o /dev/null -w 'grafana health: %{http_code}\n' http://localhost:3000/api/health
