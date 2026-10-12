# Lab Capture Guide — Generate Your Own Attack Traffic

## IMPORTANT: Safety Rules
- **ONLY** perform these attacks inside isolated VMs that you own
- **NEVER** scan or attack machines you do not own
- Use a host-only or NAT network with no internet access
- This is for educational/research purposes only

## Setup (Two VMs on VirtualBox/VMware)
1. **Attacker VM:** Kali Linux or Ubuntu with nmap and hydra installed
2. **Target VM:** Ubuntu Server with SSH (port 22) and a web server (port 80) running
3. **Network:** Host-only adapter, both VMs on the same subnet (e.g., 192.168.56.0/24)

## Step 1: Start Packet Capture on Target
```bash
# On the Target VM
sudo tcpdump -i eth0 -w /tmp/lab_capture.pcap
```

## Step 2: Generate Benign Traffic (5 minutes)
```bash
# From Attacker VM — normal browsing/SSH
curl http://192.168.56.101/
ssh user@192.168.56.101 -o StrictHostKeyChecking=no
wget http://192.168.56.101/index.html
```

## Step 3: Port Scan Attack
```bash
# From Attacker VM
nmap -sS -p 1-1000 192.168.56.101
nmap -sV 192.168.56.101
```

## Step 4: SSH Brute Force Attack
```bash
# From Attacker VM (create a small wordlist first)
echo -e "admin\nroot\nuser\ntest\npassword\n123456" > passwords.txt
hydra -l root -P passwords.txt ssh://192.168.56.101 -t 4
```

## Step 5: Stop Capture and Convert to Flows
```bash
# Stop tcpdump (Ctrl+C)
# Install cicflowmeter
pip install cicflowmeter

# Convert pcap to flows
cicflowmeter -f /tmp/lab_capture.pcap -c /tmp/lab_flows.csv
```

## Step 6: Classify with Our Models
```bash
# Copy lab_flows.csv to data/raw/lab_capture.csv
cp /tmp/lab_flows.csv "data/raw/lab_capture.csv"

# Or upload via the Streamlit demo
streamlit run demo/app.py
# Use the "Upload CSV" tab
```
