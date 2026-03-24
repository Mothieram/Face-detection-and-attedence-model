# nginx Setup Guide — FaceID Attendance System

A step-by-step guide to configure nginx as a reverse proxy with HTTPS for the face recognition attendance pipeline on Windows.

---

## Prerequisites

- nginx for Windows — [download here](https://nginx.org/en/download.html) → extract to `C:\Program Files\nginx`
- OpenSSL — bundled with Git, or [download standalone](https://slproweb.com/products/Win32OpenSSL.html)
- FastAPI server running on `http://127.0.0.1:8000`
- PowerShell running as **Administrator**

---

## Architecture

```
Mobile / Tablet (same WiFi)
        │
        │  https://192.168.1.113
        ▼
    nginx :443  ← SSL termination
        │
        │  http://127.0.0.1:8000
        ▼
    FastAPI (uvicorn)
        │
        ├── RetinaFace + AdaFace (GPU)
        ├── Qdrant + PostgreSQL
        └── Redis (Memurai)
```

---

## Step 1 — Create SSL folder

```powershell
New-Item -ItemType Directory -Force -Path "C:\nginx\ssl"
```

---

## Step 2 — Generate SSL certificate

> If `openssl` is not recognised, add Git's bundled OpenSSL to PATH first:
>
> ```powershell
> $env:PATH += ";C:\Program Files\Git\usr\bin"
> ```

```powershell
cd C:\nginx\ssl

openssl req -x509 -nodes -days 730 -newkey rsa:2048 `
  -keyout faceid.key `
  -out faceid.crt `
  -subj "/C=IN/ST=TamilNadu/L=Coimbatore/O=FaceID/CN=192.168.1.113" `
  -addext "subjectAltName=IP:192.168.1.113,DNS:localhost,DNS:faceid.local"
```

Replace `192.168.1.113` with your actual LAN IP — find it with:

```powershell
ipconfig | findstr "IPv4"
```

**Verify files were created:**

```powershell
dir C:\nginx\ssl
```

Expected output:

```
faceid.crt
faceid.key
```

---

## Step 3 — Configure nginx

Copy `nginxExample.conf` to `C:\Program Files\nginx\conf\nginx.conf`.

Replace `YOUR_LAN_IP` with your actual LAN IP (`192.168.1.113`):

```nginx
server_name  localhost faceid.local 192.168.1.113;
```

---

## Step 4 — Add Windows Firewall rule

```powershell
netsh advfirewall firewall add rule `
    name="nginx HTTPS 443" `
    dir=in `
    action=allow `
    protocol=TCP `
    localport=443
```

**Verify the rule was added:**

```powershell
netsh advfirewall firewall show rule name="nginx HTTPS 443"
```

---

## Step 5 — Free port 80 (if blocked by IIS)

> Skip this step if nginx starts without errors.
> Only needed if you see: `bind() to 0.0.0.0:80 failed`

```powershell
net stop w3svc /y
net stop http /y
```

> **Note:** This also stops Print Spooler. Restart it after nginx is running:
>
> ```powershell
> net start spooler
> ```

---

## Step 6 — Test nginx config

```powershell
cd "C:\Program Files\nginx"
.\nginx.exe -t
```

Expected output:

```
nginx: the configuration file ...nginx.conf syntax is ok
nginx: configuration file ...nginx.conf test is successful
```

---

## Step 7 — Start nginx

```powershell
Start-Process ".\nginx.exe"
```

**Verify nginx is running:**

```powershell
tasklist | findstr nginx
```

Expected output (two processes = master + worker):

```
nginx.exe    2052  Console  1  10,716 K
nginx.exe   16740  Console  1  10,976 K
```

---

## Accessing the app

| Device               | URL                                            |
| -------------------- | ---------------------------------------------- |
| Laptop               | `https://localhost`                            |
| Mobile / Tablet      | `https://192.168.1.113`                        |
| Admin first visit    | `https://192.168.1.113/?key=YOUR_ADMIN_KEY`    |
| Readonly first visit | `https://192.168.1.113/?key=YOUR_READONLY_KEY` |

> **First visit on mobile** — browser shows a certificate warning because the certificate is self-signed. Tap **Advanced → Proceed anyway**. This only happens once per device.

> After first visit the API key saves to `localStorage` — just open `https://192.168.1.113` directly on every subsequent visit.

> **Both devices must be on the same WiFi network.**

---

## nginx commands reference

| Action                      | Command                                                        |
| --------------------------- | -------------------------------------------------------------- |
| Start                       | `Start-Process ".\nginx.exe"`                                  |
| Stop                        | `.\nginx.exe -s stop`                                          |
| Reload config (no downtime) | `.\nginx.exe -s reload`                                        |
| Test config                 | `.\nginx.exe -t`                                               |
| View error log              | `Get-Content "C:\Program Files\nginx\logs\error.log" -Tail 20` |

---

## Troubleshooting

**Port 80 blocked**

```powershell
net stop w3svc /y
net stop http /y
# Then restart nginx
.\nginx.exe
```

**Certificate error**

Regenerate the certificate (Step 2), then reload nginx:

```powershell
.\nginx.exe -s reload
```

**Mobile can't reach server**

1. Make sure mobile is on the **same WiFi** as the laptop
2. Confirm firewall rule was added (Step 4)
3. Confirm nginx is running: `tasklist | findstr nginx`

**nginx won't start — stuck process**

```powershell
taskkill /F /IM nginx.exe /T
.\nginx.exe
```

**Check nginx error log**

```powershell
Get-Content "C:\Program Files\nginx\logs\error.log" -Tail 20
```

---

## Why nginx instead of ngrok

| Feature                      | ngrok                      | nginx LAN            |
| ---------------------------- | -------------------------- | -------------------- |
| Needs internet               | Yes                        | No                   |
| Camera / GPS works           | Yes                        | Yes                  |
| Speed                        | Slow (routes via internet) | Fast (direct LAN)    |
| Consistent URL               | No (changes on restart)    | Yes (always same IP) |
| Biometric data leaves office | Yes                        | No                   |
| Cost                         | Limited free tier          | Free forever         |

nginx keeps all biometric and attendance data **inside your office network** — it never touches the internet.
