import csv
import pickle
import pandas as pd
import os

def process_data(directory, validation=False):
    # open and write to train.txt first 
    with open(f"data/{directory}/train.csv", "r") as train_csv:

        train_reader = csv.reader(train_csv, delimiter=",")
        line_count = 0 

        for row in train_reader:
            if line_count != 0:
                # write lines to the train.txt file 
                with open(f"data/{directory}/train.txt", "a") as train_txt:
                    train_txt.write(row[0]+"\n")
            line_count += 1

    # open and write to test.txt 
    with open(f"data/{directory}/test.csv", "r") as test_csv:
        test_reader = csv.reader(test_csv, delimiter=",")
        line_count = 0

        for row in test_reader:
            if line_count != 0:
                # write lines to the test.txt file 
                with open(f"data/{directory}/test.txt", "a") as test_txt:
                    test_txt.write(row[0]+"\n")
            line_count += 1

    # if there is a validation file 
    if validation:
        with open(f"data/{directory}/val.csv", "r") as val_csv:
            val_reader = csv.reader(val_csv, delimiter=",")
            line_count = 0 

            for row in val_reader:
                if line_count != 0:
                    # write lines to the val.txt file 
                    with open(f"data/{directory}/val.txt", "a") as val_txt:
                        val_txt.write(row[0]+"\n")
                line_count += 1

def pickle_labels(directory, validation=False):
    # save into [train/validation/test]_labels.pickle
    # retrive labels for pickling 
    # with open(f"data/{directory}/train.csv", "r") as train_csv:
    #     train_reader = csv.reader(train_csv, delimiter=",")
    #     line_count = 0 
    #     train_labels = []
    #     for row in train_reader:
    #         if line_count != 0:
    #             # train_labels.append(int(row[0]) - 1)
    #             if row[1] == "positive":
    #                 # positives labels 1
    #                 train_labels.append(1)
    #             elif row[1] == "negative":
    #                 # negatives labeled -1
    #                 train_labels.append(-1)
    #         line_count += 1

    #     # pickle the saved labels 
    #     with open(f"data/{directory}/txt_data/train_labels.pickle", "wb") as train_labels_file:
    #         pickle.dump(train_labels, train_labels_file)

    # with open(f"data/{directory}/test.csv", "r") as test_csv:
    #     test_reader = csv.reader(test_csv, delimiter=",")
    #     line_count = 0 
    #     test_labels = []
    #     for row in test_reader:
    #         if line_count != 0:
    #             # test_labels.append(int(row[0]) - 1)
    #             if row[1] == "positive":
    #                 # positives labels 1
    #                 test_labels.append(1)
    #             elif row[1] == "negative":
    #                 # negatives labeled -1
    #                 test_labels.append(-1)
    #         line_count += 1

    #     # pickle the saved labels 
    #     with open(f"data/{directory}/txt_data/test_labels.pickle", "wb") as test_labels_file:
    #         pickle.dump(test_labels, test_labels_file)

    if validation:
        with open(f"data/{directory}/val.csv", "r") as val_csv:
            val_reader = csv.reader(val_csv, delimiter=",")
            line_count = 0 
            val_labels = []
            for row in val_reader:
                if line_count != 0:
                    if row[1] == "positive":
                        # positives labels 1
                        val_labels.append(1)
                    elif row[1] == "negative":
                        # negatives labeled -1
                        val_labels.append(-1)
                line_count += 1
            print(f"pickled labels being dumped")
            print(f"lenght of validation labels {len(val_labels)}")
            # pickle the saved labels 
            with open(f"data/{directory}/txt_data/validation_labels.pickle", "wb") as val_labels_file:
                pickle.dump(val_labels, val_labels_file)

def process_SST_sents():
    with open("./SST2-Data/SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank/datasetSentences.txt") as dataset, \
        open("./SST2-Data/SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank/datasetSplit.txt") as split:
        # print("function working")
        line_index = 0
        for line1, line2 in zip(dataset, split):
            # break at tab to get the sentence index and the sentence 
            # skip the header
            if (line_index != 0):
                # get the sentence_index and sentence
                s_index, sent = line1.split("\t")
                # get the sentence_index and the split_label from the split file 
                s_index_split, split_label = line2.split(",")
                split_label = int(split_label)
                if (s_index == s_index_split):
                    
                    print(split_label)
                    # break
                    if split_label == 1:
                        # save in train.txt 
                        print("here")
                        with open("./data/SST-2/txt_data/train.txt", "a+") as train_file:
                            train_file.write(sent)
                            # print("successful")
                    elif split_label == 2:
                        # save in train.txt 
                        with open("./data/SST-2/txt_data/test.txt", "a+") as test_file:
                            test_file.write(sent)
                    elif split_label == 3:
                        # save in train.txt 
                        with open("./data/SST-2/txt_data/val.txt", "a+") as val_file:
                            val_file.write(sent)
            line_index += 1

# def process_IMDB(directory):
#     # if text file does not exist create it 

#     with open(os.path.join(directory, "train.csv")) as train_csv:
#         # read each row at a time
#         train_reader = csv.reader(train_csv, delimiter=",")
#         line_count = 0 

#         for row in train_reader:
#             if line_count != 0:
#                 # write lines to the train.txt file 
#                 with open(f"data/{directory}/train.txt", "a") as train_txt:
#                     train_txt.write(row[2]+"\n")
#             line_count += 1

         
def main():
    # process_SST_sents()
    # process_data("IMDB", validation=True)
    pickle_labels("IMDB", validation=True)

if __name__ == "__main__":
    main()