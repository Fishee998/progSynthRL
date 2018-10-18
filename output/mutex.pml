bit turn = 0,v0 = 0,v1 = 0,try0 = 0,try1 = 0,wi0 = 0,wi1 = 0;
byte cs0 = 0,cs1 = 0;
#define mutex (cs0 + cs1 <= 1)

active proctype p()
{
	select(wi0:0..1);
	do
		::wi0 == 0->
			try0 = 1;
			select(wi0:0..1);
			turn = 1;
			do
				::turn == 0->
					if
						::v1 == 1 && turn != 0->
							v0 = 1;
					fi
				::else->break;
			od
			cs0++;
			try0 = 0;
			cs0--;
			turn = 0;
		::else->break;
	od
}

active proctype q()
{
	select(wi1:0..1);
	do
		::wi1 == 0->
			try1 = 1;
			select(wi1:0..1);
			turn = 0;
			do
				::turn == 1->
					if
						::v1 == 0 && turn != 1->
							v1 = 0;
					fi
				::else->break;
			od
			cs1++;
			try1 = 0;
			cs1--;
			turn = 0;
		::else->break;
	od
}

ltl e1{[]mutex}
ltl e2{[]((try0 == 1) -> <>(cs0 == 1))}
ltl e3{[]((try1 == 1) -> <>(cs1 == 1))}