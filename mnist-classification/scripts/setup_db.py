import argparse
import os
import yaml
from utils import connect_to_db


parser = argparse.ArgumentParser()
parser.add_argument('--config_filepath', type=str)
args = parser.parse_args()

with open(args.config_filepath, 'rb') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if not os.path.exists(config.db_path):
    os.mknod(config.db_path)

con = connect_to_db(config.db_path)
cur = con.cursor()

query = '''
CREATE TABLE IF NOT EXISTS experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epochs INTEGER,
    learning_rate REAL
)
'''

cur.execute(query)
con.commit()
con.close()
