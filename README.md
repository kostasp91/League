To start the project:

1. In python folder:
   1. Run FetchPlayersModes with full mode to create the players_data_dunkest csv file, that is going to be used for the database of the model. You have to select the number of weeks and rounds of each week that you want to load.
   2. Run prepare_dataset to create the processed_train_dataset and latest_week_data csv files, in a folder outside the League called TrainingModel.
   3. Run train_model to process the 2 files from the previous step and create the player_model.pth and scaler.pkl files (Our model that we are going to use for the predictions).
   4. Run predict_server to start the server that we are going to use for the PlayerRecommender and TeamBuilder (POST call).
  
2. In java folder:
   1. Run the TeamBuilderVer2.java following the instructions to find your team.
