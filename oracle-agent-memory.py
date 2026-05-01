''' Script to test access to Oracle Agent Memory '''

# load dependenies
from oracleagentmemory.apis.searchscope import SearchScope
from oracleagentmemory.core import OracleAgentMemory
from oracleagentmemory.core.dbschemapolicy import SchemaPolicy
from oracleagentmemory.core.embedders.embedder import Embedder
from oracleagentmemory.core.llms.llm import Llm
import oracledb
from pathlib import Path
from oci.config import from_file

# create connection object
DB_USER = ""
DB_PASSWORD = ""

# If you want to connect using your wallet, comment out the following line.
CONNECT_STRING = '()'

def return_connection():
	try:
		pool = oracledb.create_pool(
			user=DB_USER,
			password=DB_PASSWORD,
			dsn=CONNECT_STRING,
		)
		return pool
	except Exception as e:
		raise(e)
	
connection = return_connection()
print('database connection made')

# set OCI config parameters
config = from_file('~/.oci/config')
oci_key_file = str(Path(config['key_file']).expanduser())
compartment_id = config['compartment_id']

# set embedding model
oci_embedder = Embedder(model="oci/cohere.embed-english-v3.0",
						oci_compartment_id=config['compartment_id'],
						oci_region=config['region'],
						oci_user=config['user'],
						oci_fingerprint=config['fingerprint'], 
						oci_tenancy=config['tenancy'],
						oci_key_file=oci_key_file)

# set language model
oci_llm = Llm(model="oci/openai.gpt-oss-120b",
			  oci_compartment_id=config['compartment_id'],
			  oci_region=config['region'],
			  oci_user=config['user'],
			  oci_fingerprint=config['fingerprint'],
			  oci_tenancy=config['tenancy'],
			  oci_key_file=oci_key_file)

try:
	memory = OracleAgentMemory(connection=connection,
							   embedder=oci_embedder,
							   llm=oci_llm,
							   schema_policy=SchemaPolicy.CREATE_IF_NECESSARY,
							   table_name_prefix="OAM_")
	print("successfully connected to agent memory")
except Exception as e:
	raise(e)


