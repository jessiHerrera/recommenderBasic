# codigo
#Create recommendation result from trained model
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

#cargar el modelo
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/ItemKNN-Mar-04-2024_02-05-09.pth',
) 

#external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]#fist element in array is 'PAD'(default of Recbole) ->remove it
external_user_ids = dataset.id2token(dataset.uid_field, 6)[1:]#fist element in array is 'PAD'(default of Recbole) ->remove it
external_user_ids

# topK de cada user_id
topk_items = []
for internal_user_id in list(range(dataset.user_num))[1:]:
    _, topk_iid_list = full_sort_topk([internal_user_id], model, test_data, k=100, device=config['device'])
    last_topk_iid_list = topk_iid_list[-1]
    external_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
    topk_items.append(external_item_list)
print(len(topk_items))

#user_id - prediction
external_item_str = [' '.join(x) for x in topk_items]
result = pd.DataFrame(external_user_ids, columns=['user_id'])
result['prediction'] = external_item_str
result.head()