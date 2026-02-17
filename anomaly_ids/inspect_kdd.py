import pandas as pd
import json

df = pd.read_csv('training/KDDTrain.csv')

cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files',
        'is_hot_login','is_guest_login',
        'count','srv_count','serror_rate','rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate',
        'dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate']

attacks = ['neptune','smurf','portsweep','satan','back',
           'buffer_overflow','rootkit','guess_passwd','land','teardrop','normal']

for attack in attacks:
    subset = df[df['attack_class'] == attack]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n=== {attack} (n={len(subset)}) ===")
        for c in cols:
            print(f"  {c}: {row[c]}")
