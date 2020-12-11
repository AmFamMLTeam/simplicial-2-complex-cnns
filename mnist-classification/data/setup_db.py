import argparse
import os
import sys
from utils import connect_to_db


home = os.path.expanduser('~')
sys.path.append(os.path.join(home, 'sccnn-mnist-clfn'))

parser = argparse.ArgumentParser()
parser.add_argument('--db_path', help='path to database', type=str)
args = parser.parse_args()

if not os.path.exists(args.db_path):
    os.mknod(args.db_path)

con = connect_to_db(args.db_path)
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
