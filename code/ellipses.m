Ellipses=load("../data/ellipses2D.mat");
Ellipses
pause(10)
%{
we shall now plot all the variations in the given dataset
and later permute and normalize and zero mean them-up so that we 
can use them in our shape allignment alogrithm.
%}
NumOfPointSets=Ellipses.numOfPointSets
NumOfPoints=Ellipses.numOfPoints
EllipsesPointset=Ellipses.pointSets

plot(squeeze(EllipsesPointset(1,:,:)),squeeze(EllipsesPointset(2,:,:)));
pause(15)
EllipsesPointset=normalize(EllipsesPointset-mean(EllipsesPointset,2),2);
EllipsesPointset=permute(EllipsesPointset,[3,2,1])

%we select the first element as a good begining point for our estimate of
%the mean
InitialTarget=transpose(squeeze(EllipsesPointset(1,:,:)))

Shape=permute(EllipsesPointset,[1,3,2])

[AllignedShapes,MeanShape,EigenVectors,EigenValues]=Convergence(InitialTarget,Shape,NumOfPoints,NumOfPointSets)

%{
In the following we shall plot the entire pointsets in radomized colours 
which now have been alligned together.
the plot will stay for 10 seconds
%}

PlotableOutput=permute(AllignedShapes,[2,3,1])
Gx=squeeze(PlotableOutput(1,:,:))
Gy=squeeze(PlotableOutput(2,:,:))
plot(Gx,Gy);
pause(15)

%{
Now we shall observe the plot of the Eigenvalues i.e corresponding to the
modes of variation in the alligned pointsets.
the plot will stay for 10 seconds
%}
plot(EigenValues)
pause(15)

MeanVector=reshape(MeanShape,[1,2*NumOfPoints])

%{
Now we shall observe the variation of the shape w.r.t. to the FIRST mode 
annd wwe shall use a weight term of +/- 2 for the same.
The plot shall be displayed for 5 seconds
%}

Mode1NegativeB=reshape(MeanVector -2*squeeze(transpose(EigenVectors(:,1))),[2,NumOfPoints])
Mode1PositveB=reshape(MeanVector + 2*squeeze(transpose(EigenVectors(:,1))),[2,NumOfPoints])
plot(Mode1PositveB(1,:),Mode1PositveB(2,:))
pause(15)
plot(Mode1NegativeB(1,:),Mode1NegativeB(2,:))
pause(15)

%{
Now we shall observe the variation of the shape w.r.t. to the SECOND mode 
annd wwe shall use a weight term of +/- 2 for the same.
The plot shall be displayed for 5 seconds
%}

Mode2NegativeB=reshape(MeanVector -2*squeeze(transpose(EigenVectors(:,2))),[2,NumOfPoints])
Mode2PositveB=reshape(MeanVector + 2*squeeze(transpose(EigenVectors(:,2))),[2,NumOfPoints])
plot(Mode2PositveB(1,:),Mode2PositveB(2,:))
pause(15)
plot(Mode2NegativeB(1,:),Mode2NegativeB(2,:))
pause(15)

function [X] = Allign2(target,shape,NumOfPoints,NumOfPointSets)
%{
This function is for alligning 2 shapes given in terms of their pointsets
to allign them.
The shape is expected to be allignned to the shape of the targeg.

%}
    [U,~,V] = svd(shape*transpose(target));
    Vt=transpose(V)

    if(det(transpose(U*Vt)) > 0)
        X=transpose(U*Vt)*shape;
    end
    
    if(det(transpose(U*Vt)) < 0)
        DiagonalElements(1)=1;
        DiagonalElements(2)=-1;
        X=V*diag(DiagonalElements)*transpose(U)*shape;
    end
end

function Y = AllignAll(Mean,X,NumOfPoints,NumOfPointSets)
%{
This function alligns the entire dataset of shapes to a particular target
shape.
we make use of the method Allign2 to do the same.
%}
    Y=zeros(NumOfPointSets,2,NumOfPoints);
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
        AllignedShapes=AllignAll(CurrentTarget,shape,NumOfPoints,NumOfPointSets)
        NewTarget=squeeze(mean(AllignedShapes,1))
        ShapeMean=NewTarget
        [EigenVectors,~,EigenValues]=pca(reshape(AllignedShapes,[NumOfPointSets,2*NumOfPoints]))
        
        % We make our plots using the ShapeMea
        % we wait for 5 seconnds for viewing the plot of average shape in
        % our current iteration. ususally as onnly 1 or 2 iterations are
        % required this will only occur once
        
        
        PlotableOutput=ShapeMean
        Gx=squeeze(PlotableOutput(1,:))
        Gy=squeeze(PlotableOutput(2,:))
        plot(Gx,Gy);
        pause(15)
        
        
        if abs(sum(CurrentTarget-NewTarget,'all'))<0.00001
            break
        CurrentTarget=NewTarget
        
        end
    end
end  