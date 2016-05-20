function dupl= findDupl(in)
	A=zeros(size(in,1));
	for i=1:size(in,1)-1
		if in(i,1) == in(i+1,1)
			A(i)=1;
		end
	end
	dupl=find(A);
end
