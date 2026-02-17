"""
    Generate realistic test samples from actual KDD dataset
    Uses Randomization to generate random intrusion samples and normal samples 
"""
import pandas as pd
import json
import random

random.seed(42)

df_train = pd.read_csv('training/KDDTrain.csv')
df_test = pd.read_csv('training/KDDTest.csv')

# Combine for more variety
df = pd.concat([df_train, df_test])

# Fields the API expects
api_fields = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Check if is_host_login exists, if not use 0
if 'is_host_login' not in df.columns:
    df['is_host_login'] = 0

test_cases = []

# --- NORMAL SAMPLES (20 diverse cases) ---
normal_df = df[df['attack_class'] == 'normal']
normal_services = ['http', 'ftp_data', 'smtp', 'private', 'telnet', 
                   'ftp', 'ssh', 'domain_u', 'pop_3', 'other']

for i, svc in enumerate(normal_services):
    subset = normal_df[normal_df['service'] == svc]
    if len(subset) == 0:
        subset = normal_df
    row = subset.sample(1, random_state=42+i).iloc[0]
    data = {}
    for f in api_fields:
        val = row.get(f, 0)
        if isinstance(val, (int, float)):
            data[f] = float(val) if isinstance(val, float) else int(val)
        else:
            data[f] = str(val)
    test_cases.append({
        "name": f"KDD Normal {svc.upper()} Traffic #{i+1}",
        "expected": "normal",
        "data": data
    })

# Get more diverse normals
for i in range(10):
    row = normal_df.sample(1, random_state=100+i).iloc[0] # Randomizing the values of novel attacks
    data = {}
    for f in api_fields:
        val = row.get(f, 0)
        if isinstance(val, (int, float)):
            data[f] = float(val) if isinstance(val, float) else int(val)
        else:
            data[f] = str(val)
    test_cases.append({
        "name": f"KDD Normal Traffic (random) #{i+1}",
        "expected": "normal",
        "data": data
    })

# --- ATTACK SAMPLES (from known attack classes) ---
attack_types = {
    'neptune': 'DoS SYN Flood (neptune)',
    'smurf': 'DoS ICMP Flood (smurf)',
    'satan': 'Probe - Satan Scanner',
    'ipsweep': 'Probe - IP Sweep',
    'portsweep': 'Probe - Port Sweep',
    'nmap': 'Probe - Nmap Scan',
    'back': 'DoS - Back (Apache)',
    'teardrop': 'DoS - Teardrop',
    'pod': 'DoS - Ping of Death',
    'warezclient': 'R2L - Warez Client',
    'guess_passwd': 'R2L - Password Guessing',
    'buffer_overflow': 'U2R - Buffer Overflow',
    'rootkit': 'U2R - Rootkit',
    'land': 'DoS - Land Attack',
    'loadmodule': 'U2R - Loadmodule',
    'ftp_write': 'R2L - FTP Write',
    'imap': 'R2L - IMAP Attack',
    'multihop': 'R2L - Multihop',
    'warezmaster': 'R2L - Warez Master',
}

for attack_name, label in attack_types.items():
    subset = df[df['attack_class'] == attack_name]
    if len(subset) == 0:
        continue
    # Take up to 3 samples per attack type
    n_samples = min(3, len(subset))
    for j in range(n_samples):
        row = subset.sample(1, random_state=200+j).iloc[0]
        data = {}
        for f in api_fields:
            val = row.get(f, 0)
            if isinstance(val, (int, float)):
                data[f] = float(val) if isinstance(val, float) else int(val)
            else:
                data[f] = str(val)
        suffix = f" #{j+1}" if n_samples > 1 else ""
        test_cases.append({
            "name": f"KDD Attack: {label}{suffix}",
            "expected": "intrusion",
            "data": data
        })

# --- NOVEL ATTACKS (from test set only) ---
test_only_attacks = df_test[~df_test['attack_class'].isin(df_train['attack_class'].unique())] # Novel attacks indexes
novel_types = test_only_attacks['attack_class'].unique() # Actual Novel Attack values

for attack_name in novel_types[:8]:  # First 8 novel attacks
    subset = test_only_attacks[test_only_attacks['attack_class'] == attack_name]
    if len(subset) == 0:
        continue
    row = subset.sample(1, random_state=300).iloc[0]
    data = {}
    for f in api_fields:
        val = row.get(f, 0)
        if isinstance(val, (int, float)):
            data[f] = float(val) if isinstance(val, float) else int(val)
        else:
            data[f] = str(val)
    test_cases.append({
        "name": f"KDD Novel Attack: {attack_name}",
        "expected": "intrusion",
        "data": data
    })

# Save
output = {"test_cases": test_cases}
with open('test_samples.json', 'w') as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(test_cases)} test cases")
print(f"  Normal: {sum(1 for t in test_cases if t['expected'] == 'normal')}")
print(f"  Intrusion: {sum(1 for t in test_cases if t['expected'] == 'intrusion')}")
print(f"\nAttack types covered:")
for t in test_cases:
    if t['expected'] == 'intrusion':
        print(f"  - {t['name']}")
