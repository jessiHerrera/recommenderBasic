from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from tqdm.auto import tqdm
import pandas as pd

k = 100
#modelf = 'ItemKNN-Apr-05-2024_01-00-27.pth'
#modelf = 'ItemKNN-Apr-12-2024_16-54-50.pth'
#modelf = 'BPR-Apr-12-2024_17-05-56.pth'
#modelf = 'BPR-Apr-12-2024_17-31-14.pth'
#modelf = 'ItemKNN-Apr-12-2024_18-56-49.pth'
#modelf = 'BPR-Apr-12-2024_18-57-46.pth'
#modelf = 'ItemKNN-Apr-12-2024_19-20-33.pth'
#modelf = 'BPR-Apr-12-2024_19-22-48.pth'
#15 mayo
modelf = 'ItemKNN-May-15-2024_23-43-37.pth'

if __name__ == '__main__':

    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/ItemKNN-Apr-05-2024_01-00-27.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/ItemKNN-Apr-12-2024_16-54-50.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/BPR-Apr-12-2024_17-05-56.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/BPR-Apr-12-2024_17-31-14.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/ItemKNN-Apr-12-2024_18-56-49.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/BPR-Apr-12-2024_18-57-46.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/ItemKNN-Apr-12-2024_19-20-33.pth')
    #config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/BPR-Apr-12-2024_19-22-48.pth')
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file='D:/pythonProjects/JupyterProjects/recbole/saved/ItemKNN-May-15-2024_23-43-37.pth')

    ground_list = []
    uid_list = []

    for batch_idx, batched_data in enumerate(test_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        ground_list.append([int(v) for v in positive_i.numpy().tolist()])
        uid_list.append(interaction.user_id.numpy()[0])
        
    external_user_ids = dataset.id2token(test_data.uid_field, list(range(dataset.user_num)))[1:]#fist element in array is 'PAD'(default of Recbole) ->remove it
    external_user_ids
    
    ranked_list = []
    topk_items = []
    topk_scores = []
    external_user_list = []
    external_item_list = []
        
    for uid in tqdm(uid_list):
        topk_score, topk_iid_list = full_sort_topk([uid], model, test_data, k, device=config['device'])
        ranked_list += topk_iid_list.cpu()
        last_topk_iid_list = topk_iid_list[-1]
        last_topk_score = topk_score[-1]
        external_user_list = external_user_ids.tolist()
        external_item_list = dataset.id2token(test_data.iid_field, last_topk_iid_list.cpu()).tolist()
        external_score_list = last_topk_score.cpu().tolist()    
        topk_items.append(external_item_list)
        topk_scores.append(external_score_list)    


    print(len(topk_items))

    df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/datos_reducidos_choice_100000.csv")
    #df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/datos_reducidos_choiceProb_100000.csv")
    #df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/datos_reducidos_Sample_100000.csv")
    #df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/datos_reducidos_normal_100000.csv")

    items_r = []
    for row in topk_items:
        for element in row:
            items_r.append(element)
            
    score_r = []
    for row in topk_scores:
        for elements2 in row:
            score_r.append(float(elements2))
            

    resultado = []

    for v in uid_list:
        l = [v] * k
        resultado = resultado + l

    resultF1 = pd.DataFrame(resultado, columns=['user_id'])
    resultF1['item_rec']= items_r
    resultF1['score_rec']= score_r

    resultF1.to_csv(f"D:/pythonProjects/JupyterProjects/data/bgg/txt/resulF2_{modelf}_{k}.csv", index=False, sep=';')
    resultF1.to_csv('D:/pythonProjects/JupyterProjects/elliot_env/data/elliot/txt/predictions_n.tsv',  sep='\t', header=None, index=False,)

