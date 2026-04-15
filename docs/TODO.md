# AetherForecast TODO

## Priority 1
- [x] Toi uu Docker image backend
  - Multi-stage build: builder + runtime
  - Runtime base image: python:3.11-slim
  - Runtime chi giu dependency cho FastAPI, Caddy, inference, websocket, cron
  - Loai training dependency trong runtime image (datasets, peft, scipy...)
  - Ep torch CPU wheel de giam kich thuoc image
  - Cap nhat .dockerignore de loai bo code training khoi build context

## Next
- [ ] Deploy stack sau khi xac nhan image size moi
- [ ] Cai GitHub CLI (gh) va set repository variables/secrets
