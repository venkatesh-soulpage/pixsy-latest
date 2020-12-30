import dask.dataframe as dd

m_df= dd.read_csv('match_features_local_global/0.part')
m_df = m_df.drop(['Unnamed: 0'],axis='columns')
p_df= dd.read_csv('photo_features_local_global/0.part')
p_df = p_df.drop(['Unnamed: 0'],axis='columns')

fp_df = dd.read_csv('false_positives_dataset.csv')
fp_df = fp_df[['match_id','photo_id','match_pixsy_label']]
fp_df = fp_df.rename(columns={'match_pixsy_label':'FP'})
fp_df['FP'] = fp_df['FP'].mask(fp_df['FP']==10,True).mask(fp_df['FP']==11,False)
fp_photo_match_feat =fp_df.merge(p_df,how='inner',on='photo_id').merge(m_df,how='inner',on='match_id',suffixes=('_photo','_match'))

tp_df = dd.read_csv('true_positives_dataset.csv')
tp_df = tp_df[['match_id','photo_id','match_fp']]
tp_df = tp_df.rename(columns={'match_fp':'FP'})
tp_df['FP'] = tp_df['FP'].mask(tp_df['FP']==True,False)
tp_photo_match_feat =tp_df.merge(p_df,how='inner',on='photo_id').merge(m_df,how='inner',on='match_id',suffixes=('_photo','_match'))

final_data_for_classification = dd.concat([fp_photo_match_feat,tp_photo_match_feat])
print(len(final_data_for_classification))
final_data_for_classification.to_csv('pixsy_data_for_classification')
