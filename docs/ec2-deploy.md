# AWS EC2 deploy — `abol-mms-stt`

Self-hosted Meta MMS ASR on Ubuntu + Docker Compose.  
Expose **80/443** only. Keep **8000** bound to localhost.

## Instance sizing

| Instance | Device | Expectation |
|----------|--------|-------------|
| `t3.xlarge` (4 vCPU, 16 GiB) | CPU | Works, **slow**. OK for short demos. |
| `g4dn.xlarge` (1× T4, 16 GiB) | CUDA | **Better** latency. Prefer for real use. |

Root volume: **≥ 40 GiB** (model cache is several GB).

Stop when idle:

```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxx
```

## 1. Security group

| Inbound | Source |
|---------|--------|
| TCP 22 | your IP |
| TCP 80 | `0.0.0.0/0` (or tighter) |
| TCP 443 | `0.0.0.0/0` (or tighter) |

Do **not** open TCP 8000.

## 2. Ubuntu host prep

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# log out and back in so the docker group applies
docker version
docker compose version || docker-compose version
```

### GPU only (`g4dn.*`) — recommended for full podcasts

```bash
# NVIDIA driver (Ubuntu) — reboot if prompted
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# then:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

```bash
cp .env.example .env
# DEVICE=cuda
sed -i 's/^DEVICE=.*/DEVICE=cuda/' .env

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
until curl -fsS http://127.0.0.1:8000/ready; do sleep 5; done

# Full 22-min episode example
curl -sS -X POST \
  -F "file=@./tests/bitcoin-revolution-ethiopia-podcast-clip.ogg;type=audio/ogg" \
  "http://127.0.0.1:8000/transcribe?lang=am" \
  -o /tmp/mms-bitcoin-full-am.json
```

Stop the instance when idle to save cost.

## 3. Clone and configure

```bash
git clone https://github.com/ibo-123/Abol-kit-service.git abol-mms-stt
cd abol-mms-stt
cp .env.example .env
# nano .env   # DEVICE=cpu|cuda, MAX_UPLOAD_MB=50, optional API_KEY=
```

## 4. Run the wrapper

```bash
docker compose up --build -d
# older Engine without the compose plugin:
# docker-compose up --build -d

docker compose logs -f mms-stt
# docker-compose logs -f mms-stt
```

Compose publishes **`127.0.0.1:8000:8000`** only (intentional).

```bash
# Process up (ready may still be false while weights download)
curl -s http://127.0.0.1:8000/health
# {"ok":true,"ready":false|true,"model":"facebook/mms-1b-all","version":"0.1.0"}

# Block until model is loaded (first boot can take several minutes)
until curl -fsS http://127.0.0.1:8000/ready; do sleep 5; done
```

Weights land in the Docker volume `hf-cache` and survive restarts.

## 5. Reverse proxy (80/443 → 127.0.0.1:8000)

### Caddy (TLS automatic)

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy
```

`/etc/caddy/Caddyfile`:

```caddyfile
stt.example.com {
  reverse_proxy 127.0.0.1:8000
}
```

```bash
sudo systemctl reload caddy
curl -fsS https://stt.example.com/health
```

### nginx sketch

```nginx
server {
  listen 443 ssl http2;
  server_name stt.example.com;
  # ssl_certificate /etc/letsencrypt/live/stt.example.com/fullchain.pem;
  # ssl_certificate_key /etc/letsencrypt/live/stt.example.com/privkey.pem;
  client_max_body_size 50m;

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
  }
}
```

## 6. Point Abol Kit at the service

```bash
# Same Docker network (best on one host / shared compose network)
TRANSCRIBER_URL=http://mms-stt:8000

# Via public hostname (proxy → localhost:8000)
TRANSCRIBER_URL=https://stt.example.com
```

Adapter wiring: [typescript-client.md](./typescript-client.md).

## 7. Smoke test on the host

```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh en
./scripts/smoke_test.sh am /path/to/amharic.mp3
```

## 8. Idle cost control

1. Stop Abol Kit jobs that call STT.  
2. Optional: `docker compose down` (volume keeps the model cache).  
3. **Stop** the EC2 instance.  
4. Start later — no re-download if the EBS volume (and `hf-cache` volume) persist.

## Privacy

- Self-hosted default: `sendsDataOffHost: false`  
- No third-party STT; no API key unless you set `API_KEY`  
- Audio stays on-box unless you put a public proxy in front
