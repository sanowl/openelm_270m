
from transformers import AutoModelForCausalLM

openelm_270m = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True)
openelm_450m = AutoModelForCausalLM.from_pretrained("apple/OpenELM-450M", trust_remote_code=True)
openelm_1b = AutoModelForCausalLM.from_pretrained("apple/OpenELM-1_1B", trust_remote_code=True)
openelm_3b = AutoModelForCausalLM.from_pretrained("apple/OpenELM-3B", trust_remote_code=True)

openelm_270m_instruct = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)
openelm_450m_instruct = AutoModelForCausalLM.from_pretrained("apple/OpenELM-450M-Instruct", trust_remote_code=True)
openelm_1b_instruct = AutoModelForCausalLM.from_pretrained("apple/OpenELM-1_1B-Instruct", trust_remote_code=True)
openelm_3b_instruct = AutoModelForCausalLM.from_pretrained("apple/OpenELM-3B-Instruct", trust_remote_code=True)
