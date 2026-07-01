% in this version we perform extrapolation
function z=smooth2(x,y,delta)

sx=size(x);
if(sx(1)>1);  xtemp=x'; else ; xtemp=x; end ;

nx=length(x);
z=zeros(1,nx);

for n=1:delta
% correction for external points
  tempx=xtemp(n:delta:nx);% sx=size(tempx);if(sx(1)>1); tempx=tempx';end;
  tempy=y(n:delta:nx);sy=size(tempy);if(sy(1)>1); tempy=tempy';end;
  
  
  if (tempx(1)>x(1))
   tempy=[tempy(1)+(tempy(2)-tempy(1))/(tempx(2)-tempx(1))*(x(1)-tempx(1)) tempy];tempx=[x(1) tempx];
  end  
  if (tempx(end)<x(end))
   tempy=[ tempy tempy(end)+(tempy(end)-tempy(end-1))/(tempx(end)-tempx(end-1))*(x(end)-tempx(end)) ];tempx=[tempx x(end) ];
  end  
  
  z=z+interp1(tempx,tempy,xtemp);
end

z=z./delta;
% restore dimension order
if(sx(1)>1);  z=z'; end ;

return;

