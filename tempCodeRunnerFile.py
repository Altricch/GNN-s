            
            # print("batch mask", np.where(batch.train_mask == True))
            print(pred[batch.train_mask])
            # pred = torch.argmax(pred[batch.train_mask], dim=1)