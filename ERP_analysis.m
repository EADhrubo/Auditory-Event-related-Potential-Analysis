%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%      ERP extraction      %%%%%%

clear all;
clc;
close all;

load EEGdata2.mat

Fs = 100;

% spatial filter
for ch = 1:16
    temp = Edat;
    temp(:,ch,:,:)=[];
    fdat(:,ch,:,:) = Edat(:,ch,:,:)- mean(temp,2);
end


%% raw average CCA
clear all;
clc;
close all;

load EEGdata2.mat
nrep = 20;
nfold = 10;
ground = [];
pred = [];
for fold = 1: nfold
    for i = 1:nrep
        aa = randperm(30, 15);
        dd = mean(Edat(:,:,aa,:),3);
        newdat(:,:,i,:) = dd;
    end

    for rep = 1:nrep
        temp = newdat;
        temp(:,:,rep,:)=[];
        testdat = squeeze(newdat(:,:,rep,:));
        traindat = squeeze(mean(temp,3));
        for tsk1 = 1:6
            for tsk2 = 1:6
                x = testdat(:,:,tsk1)';
                y = traindat(:,:,tsk2)';
                [A,B,r] = canoncorr(x,y);
                rr(tsk2) = mean([corr2(x*A,y*A),corr2(x*B,y*B)]);
            end
            [v,loc(tsk1)] = max(rr);
        end
        confus_max(fold,rep,:) = ones(1,6) .* (loc==[1,2,3,4,5,6]);
        acc(fold,rep) = length(find(loc==[1,2,3,4,5,6]));
        truth = [1, 2, 3, 4, 5, 6];
        ground = [ground truth];
        pred = [pred loc];
    end
end

Accuracy = sum(acc,"all")/(6*nrep*nfold)



%% AR + LDA

clear all;
clc;
close all;

load EEGdata2.mat

% % spatial filter
% for ch = 1:16
%     temp = Edat;
%     temp(:,ch,:,:)=[];
%     fdat(:,ch,:,:) = Edat(:,ch,:,:)- mean(temp,2);
% end

nrep = 30;
nfold = 10;
ground = [];
pred = [];

for fold = 1:nfold
    for i = 1:nrep
        aa = randperm(30,15);
        dd = mean(Edat(:,:,aa,:),3);
        newdat(:,:,i,:) = dd;
    end

    Fs = 100;
    for tsk = 1:6
        for rep = 1:nrep
            for ch = 1:16
                pxx = ar(newdat(10:80,ch,rep,tsk),6,'gl','now','Ts',0.01);
                fea(:,ch,rep,tsk) = pxx.A(2:end);
            end
        end
    end



    aa = reshape(randperm(nrep),5,6);
    bb = aa;
    bb(1,:) = [];
    tt = aa(1,:);
    tr = bb(:);
    lr = length(tr);
    lt = length(tt);


    train = reshape(fea(:,:,tr,:),size(fea,1)*size(fea,2),lr*6)';
    test = reshape(fea(:,:,tt,:),size(fea,1)*size(fea,2),lt*6)';

    for i = 1:6
        j = (i-1)*lr+1 : i*lr;
        k = (i-1)*lt+1 : i*lt;
        lbl_tr(j,:) = i;
        lbl_tt(k,:) = i;
    end

    for pc = 46
        coeff = pca(train);
        pctrain = train*coeff(:,1:pc);
        pctest = test*coeff(:,1:pc);

        Mdl = fitcdiscr(pctrain,lbl_tr);
        [y,score] = predict(Mdl,pctest);

        acc(fold,pc) = length(find(y==lbl_tt))/length(y);
    end

    ground = [ground lbl_tt'];
    pred = [pred y'];
    accuracy(1,fold) = acc(fold,pc);
end
Accuracy = sum(accuracy) / nfold

%plot(acc)
%xlabel('Number of PCs')
%ylabel('Classification Accuracy of testing set')
%C = confusionmat(ground,pred)
%cm = confusionchart(ground,pred,'RowSummary','row-normalized','ColumnSummary','column-normalized')
% xlabel('Number of AR Order', 'FontSize', 20)
%ylabel('Classification Accuracy of testing set','FontSize',20)
%plot(x(6), y(6), 'or', 'LineWidth', 2);
%caption = sprintf('x=%.2f, y=%.2f', x(6), y(6))
%text(x(6) + 0.4, y(6) + 0.02, caption,"FontSize",20)


%% Plot of ERPs
clear all;
clc;
close all;

load EEGdata2.mat

ss = {'Crash','Camera','Car horn','Lock','Dog bark','Gun shot'};


for tsk = 1:6
    aa = mean(Edat(1:60,8,:,tsk),3);
    subplot(2,3,tsk),plot(aa);
    title(ss{tsk},'Fontsize',12)
end




