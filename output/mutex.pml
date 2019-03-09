bit turn = 0,v0 = 0,v1 = 0,v2 = 0;
bit t0 = 0,t1 = 0,t2 = 0;
bit e0 = 0,e1 = 0,e2 = 0;
bit wi0 = 0,wi1 = 0,wi2 = 0;
int s0 = -1,s1 = -1,s2 = -1,s3 = -1;
active proctype p0()
{
	select(wi0:0..1);
	do
		::wi0 == 0->
			select(wi0:0..1);
			t0 = 1;
			t0 = 0;
			if
				::v1 != 1->
			fi
			e0 = 1;
			e0 = 0;
			if
				::v2 == 0->
					v0 = 1;
			fi
			atomic{s1==0->s1 = -1;}
		::else->break;
	od
}

active proctype p1()
{
	select(wi0:0..1);
	do
		::wi1 == 0->
			select(wi1:0..1);
			t1 = 1;
			t1 = 0;
			if
				::v2 != 1->
			fi
			e1 = 1;
			e1 = 0;
			if
				::v0 == 0->
					v1 = 1;
			fi
			atomic{s2==1->s2 = -1;}
		::else->break;
	od
}

active proctype p2()
{
	select(wi0:0..1);
	do
		::wi2 == 0->
			select(wi2:0..1);
			t2 = 1;
			t2 = 0;
			if
				::v0 != 1->
			fi
			e2 = 1;
			e2 = 0;
			if
				::v1 == 0->
					v2 = 1;
			fi
			atomic{s0==2->s0 = -1;}
		::else->break;
	od
}

ltl e1{[]((t0 == 1) -> <>(e0 == 1))}
ltl e2{[]((e0 == 1) -> ((s0 == 0) && (s1 == 0)))}
ltl e3{[]((t1 == 1) -> <>(e1 == 1))}
ltl e4{[]((e1 == 1) -> ((s1 == 1) && (s2 == 1)))}
ltl e5{[]((t2 == 1) -> <>(e2 == 1))}
ltl e6{[]((e2 == 1) -> ((s2 == 2) && (s0 == 2)))}
