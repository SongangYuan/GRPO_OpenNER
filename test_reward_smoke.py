from importlib.machinery import SourceFileLoader
import os

# Load custom_reward.py directly from the current folder
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'custom_reward.py'))
cr = SourceFileLoader('custom_reward', module_path).load_module()

# Sample solution and ground truths
solution_ok = (
    '<analysis>\n- The text mentions "using a Mac" which refers to Apple Macintosh computer.\n</analysis>'
    '<ner_result>["Mac"]</ner_result>'
)
solution_empty = '<analysis>The entity is absent.</analysis><ner_result>[]</ner_result>'

gt_list = ["Mac"]
gt_dict = {"entities": ["Mac"]}

print('--- smoke: correct prediction vs list gt ---')
print(cr.compute_ner_score(solution_ok, gt_list, data_source='ner_dapo', extra_info={"batch_idx":0, "sample_idx":0}))

print('--- smoke: empty prediction vs list gt ---')
print(cr.compute_ner_score(solution_empty, gt_list, data_source='ner_dapo', extra_info={"batch_idx":1, "sample_idx":0}))

print('--- smoke: correct prediction vs dict gt ---')
print(cr.compute_ner_score(solution_ok, gt_dict, data_source='ner_dapo', extra_info={"batch_idx":2, "sample_idx":0}))