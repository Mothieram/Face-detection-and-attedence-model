# Nginx Setup Guide - FaceID Attendance System (Windows)

This guide is for developers who receive this project and need to run the API behind Nginx with HTTPS on LAN.

It includes:

- Step-by-step setup
- How to verify each step
- What success output should look like
- How to validate API behavior through Nginx

---

## 1. What This Setup Does

Nginx terminates HTTPS and forwards all requests to FastAPI:

```text
Mobile/Browser -> https://<LAN_IP>:443 (Nginx) -> http://127.0.0.1:8000 (FastAPI)
```

This gives:

- Same-LAN access from phones/tablets
- HTTPS URL for camera/GPS browser behavior
- Stable local endpoint without ngrok

---

## 2. Prerequisites

- Nginx for Windows installed at `C:\Program Files\nginx`
- OpenSSL available (Git OpenSSL or Win64 OpenSSL)
- FastAPI running on `http://127.0.0.1:8000`
- Admin PowerShell

Recommended project files:

- Config template: `nginx/nginxexample.conf`
- Setup guide: `nginx/nginxsetup.md`

---

## 3. Step-by-Step Setup

## Step 1: Find your LAN IP

```powershell
ipconfig | findstr "IPv4"
```

Example:

```text
IPv4 Address. . . . . . . . . . . : 192.168.1.113
```

Keep this value. We will use `<LAN_IP>` in the next steps.

---

## Step 2: Create SSL directory

```powershell
New-Item -ItemType Directory -Force -Path "C:\nginx\ssl"
```

Verify:

```powershell
Test-Path "C:\nginx\ssl"
```

Expected output:

```text
True
```

---

## Step 3: Generate self-signed certificate

If `openssl` is not recognized:

```powershell
$env:PATH += ";C:\Program Files\Git\usr\bin"
```

Generate cert/key (replace `<LAN_IP>`):

```powershell
cd C:\nginx\ssl

openssl req -x509 -nodes -days 730 -newkey rsa:2048 `
  -keyout faceid.key `
  -out faceid.crt `
  -subj "/C=IN/ST=TamilNadu/L=Coimbatore/O=FaceID/CN=<LAN_IP>" `
  -addext "subjectAltName=IP:<LAN_IP>,DNS:localhost,DNS:faceid.local"
```

Verify files:

```powershell
dir C:\nginx\ssl
```

Expected:

```text
faceid.crt
faceid.key
```

---

## Step 4: Configure Nginx

1. Open template: `nginx/nginxexample.conf`
2. Replace `YOUR_LAN_IP` in:

```nginx
server_name localhost faceid.local YOUR_LAN_IP;
```

3. Copy to active config:

```powershell
Copy-Item `
  -Path "E:\django\bookstore\nginx\nginxexample.conf" `
  -Destination "C:\Program Files\nginx\conf\nginx.conf" `
  -Force
```

Notes:

- Template has `client_max_body_size 10M` (matches FastAPI upload limit).
- HTTP->HTTPS redirect block is commented by default; HTTPS on 443 still works.

---

## Step 5: Open firewall port 443

```powershell
netsh advfirewall firewall add rule `
  name="nginx HTTPS 443" `
  dir=in `
  action=allow `
  protocol=TCP `
  localport=443
```

Verify:

```powershell
netsh advfirewall firewall show rule name="nginx HTTPS 443"
```

---

## Step 6: Test Nginx configuration

```powershell
cd "C:\Program Files\nginx"
.\nginx.exe -t
```

Expected:

```text
nginx: the configuration file ...nginx.conf syntax is ok
nginx: configuration file ...nginx.conf test is successful
```

---

## Step 7: Start or reload Nginx

First start:

```powershell
Start-Process ".\nginx.exe"
```

If already running (after config edits):

```powershell
.\nginx.exe -s reload
```

Verify running:

```powershell
tasklist | findstr nginx
```

Expected: two nginx processes (master + worker).

---

## 4. API Validation Through Nginx

After Nginx starts, validate that proxy works and API responses pass through correctly.

## Check API status

```powershell
curl.exe -k https://<LAN_IP>/v1/status -H "X-API-Key: YOUR_KEY"
```

Expected JSON keys:

```json
{
  "cvlface_loaded": true,
  "liveness_enabled": true,
  "active_liveness_enabled": true,
  "db_tier": {},
  "match_thresholds": {}
}
```

## Check docs page

Open:

```text
https://<LAN_IP>/docs
```

If page opens, Nginx routing to FastAPI is working.

---

## 5. Access URLs

| Device                    | URL                                       |
| ------------------------- | ----------------------------------------- |
| Laptop                    | `https://localhost`                       |
| Mobile/Tablet (same WiFi) | `https://<LAN_IP>`                        |
| Admin first visit         | `https://<LAN_IP>/?key=YOUR_ADMIN_KEY`    |
| Readonly first visit      | `https://<LAN_IP>/?key=YOUR_READONLY_KEY` |

Notes:

- First visit may show cert warning (self-signed). Accept and continue.
- API key is stored in browser `localStorage` after first key-based visit.

---

## 6. Command Reference

| Action         | Command                                                        |
| -------------- | -------------------------------------------------------------- |
| Test config    | `.\nginx.exe -t`                                               |
| Start          | `Start-Process ".\nginx.exe"`                                  |
| Reload         | `.\nginx.exe -s reload`                                        |
| Stop           | `.\nginx.exe -s stop`                                          |
| View error log | `Get-Content "C:\Program Files\nginx\logs\error.log" -Tail 50` |

---

## 7. Troubleshooting

## Nginx not starting

```powershell
Get-Content "C:\Program Files\nginx\logs\error.log" -Tail 50
```

## Port 443 blocked

Re-check firewall rule:

```powershell
netsh advfirewall firewall show rule name="nginx HTTPS 443"
```

## Port 80 conflict appears

Only relevant if you enabled HTTP redirect block on port 80.

If needed:

```powershell
net stop w3svc /y
net stop http /y
```

## Mobile cannot connect

1. Confirm mobile and server are on same WiFi.
2. Confirm Nginx process exists (`tasklist | findstr nginx`).
3. Confirm API responds locally:

```powershell
curl.exe -k https://localhost/v1/status -H "X-API-Key: YOUR_KEY"
```

## Certificate mismatch warning

Regenerate cert with correct current LAN IP in both:

- `CN=<LAN_IP>`
- `subjectAltName=IP:<LAN_IP>`

Then reload:

```powershell
.\nginx.exe -s reload
```

---

## 8. Security Note

This is a LAN-oriented setup with a self-signed certificate.  
For internet exposure, use:

- valid CA certificate
- proper DNS
- hardened firewall policy
- optional VPN or zero-trust gateway
