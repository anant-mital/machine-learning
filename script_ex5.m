tr_labels_nn=zeros(50000,10);
tr_labels=double(tr_labels);
tr_data=double(tr_data);
te_data=double(te_data);
te_labels=double(te_labels);
te_data_nn=transpose(te_data);
for i=1:50000
    for j=1:10
        if tr_labels(i,1)==j-1
            tr_labels_nn(i,j)=1;
        end
    end
end
neuralnet=cifar_10_MLP_train(tr_data,tr_labels_nn);
pred=cifar_10_MLP_test(te_data_nn,neuralnet);
network_accuracy=cifar_10_evaluate(pred,te_labels);
