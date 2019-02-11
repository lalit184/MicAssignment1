Bone=load("../data/bone3D.mat");
BoneData=Bone.shapesTotal
BoneData=permute(BoneData,[3,1,2])
%PlotAllData(BoneData,30)
%title("crap")
%PlotAllData(mean(BoneData,1),1)



%{
we shall now plot all the variations in the given dataset
and later permute and normalize and zero mean them-up so that we 
can use them in our shape allignment alogrithm.
%}

NumOfPointSets=30
NumOfPoints=252

BoneData=normalize(BoneData-mean(BoneData,3),3);
%PlotAllData(BoneData,30)
size(mean(BoneData,1))
pause(2)


%{
we select the first element as a good begining point for our estimate of
the mean
%}

InitialTarget=squeeze(BoneData(1,:,:))

Shape=BoneData

[AllignedShapes,MeanShape,EigenVectors,EigenValues]=Convergence(InitialTarget,Shape,NumOfPoints,NumOfPointSets)
%PlotAllData(AllignedShapes,30)

%{
In the following we shall plot the entire pointsets in radomized colours 
which now have been alligned together.
the plot will stay for 10 seconds
%}
%{
PlotableOutput=permute(AllignedShapes,[2,3,1])
Gx=squeeze(PlotableOutput(1,:,:))
Gy=squeeze(PlotableOutput(2,:,:))
plot(Gx,Gy);
pause(15)
%}
%{
Now we shall observe the plot of the Eigenvalues i.e corresponding to the
modes of variation in the alligned pointsets.
the plot will stay for 10 seconds
%}
%plot(EigenValues)
%pause(15)

MeanVector=reshape(MeanShape,[1,3*NumOfPoints]);

%{
Now we shall observe the variation of the shape w.r.t. to the FIRST mode 
annd wwe shall use a weight term of +/- 2 for the same.
The plot shall be displayed for 5 seconds
%}

Mode1NegativeB=reshape(MeanVector -0*squeeze(transpose(EigenVectors(:,1))),[1,3,NumOfPoints]);
Mode1PositiveB=reshape(MeanVector + 2*squeeze(transpose(EigenVectors(:,1))),[1,3,NumOfPoints]);

PlotAllData(Mode1PositiveB,1);
PlotAllData(Mode1NegativeB,1);

Mode2NegativeB=reshape(MeanVector -2*squeeze(transpose(EigenVectors(:,2))),[1,3,NumOfPoints]);
Mode2PositiveB=reshape(MeanVector + 2*squeeze(transpose(EigenVectors(:,2))),[1,3,NumOfPoints]);

PlotAllData(Mode2PositiveB,1);
PlotAllData(Mode2NegativeB,1);

function PlotAllData(BoneData,elems)
    for i=1:elems
        x1=squeeze(BoneData(i,1,:));
        y1=squeeze(BoneData(i,2,:));
        z1=squeeze(BoneData(i,3,:));
        scatter3(x1, y1, z1, 'filled')
        hold on
    end
    hold off
    pause(10)
end


function [X] = Allign2(target,shape,NumOfPoints,NumOfPointSets)
%{
This function is for alligning 2 shapes given in terms of their pointsets
to allign them.
The shape is expected to be allignned to the shape of the targeg.

%}  
    size(shape);
    size(target);
    [U,~,V] = svd(shape*transpose(target));
    Vt=transpose(V);

    if(det(transpose(U*Vt)) > 0)
        X=transpose(U*Vt)*shape;
    end
    
    if(det(transpose(U*Vt)) < 0)
        DiagonalElements(1:2)=1;
        DiagonalElements(3)=-1;
        X=V*diag(DiagonalElements)*transpose(U)*shape;
    end
end

function Y = AllignAll(Mean,X,NumOfPoints,NumOfPointSets)
%{
This function alligns the entire dataset of shapes to a particular target
shape.
we make use of the method Allign2 to do the same.
%}
    Y=zeros(NumOfPointSets,3,NumOfPoints);
    for i=1:NumOfPointSets
        Y(i,:,:)=Allign2(Mean,squeeze(X(i,:,:)));
    end
end

function [AllignedShapes,ShapeMean,EigenVectors,EigenValues] = Convergence(target,shape,NumOfPoints,NumOfPointSets)
%{
This method basically tries to allign the enitre dataset to a target shape 
and compute the neww target shape to be the arithmetic mean of all the
given shapes.
the algorithms runs till the new target computed after alliging all shaps to the current target does 
not differ to the current target shape by alot.(L1 norm is less than some threshold for these 2 tensors.
%}
    CurrentTarget=target
    while 1
        AllignedShapes=AllignAll(CurrentTarget,shape,NumOfPoints,NumOfPointSets);
        NewTarget=squeeze(mean(AllignedShapes,1));
        ShapeMean=NewTarget;
        size(AllignedShapes);
        [EigenVectors,~,EigenValues]=pca(reshape(AllignedShapes,[NumOfPointSets,3*NumOfPoints]));
        
        % We make our plots using the ShapeMea
        % we wait for 5 seconnds for viewing the plot of average shape in
        % our current iteration. ususally as onnly 1 or 2 iterations are
        % required this will only occur once
        
        
        %PlotAllData(ShapeMean,1)
        
        
        
        if abs(sum(CurrentTarget-NewTarget,'all'))<0.00001
            break
        CurrentTarget=NewTarget;
        
        end
    end
end  
%}

