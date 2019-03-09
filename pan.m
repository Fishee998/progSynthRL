#define rand	pan_rand
#define pthread_equal(a,b)	((a)==(b))
#if defined(HAS_CODE) && defined(VERBOSE)
	#ifdef BFS_PAR
		bfs_printf("Pr: %d Tr: %d\n", II, t->forw);
	#else
		cpu_printf("Pr: %d Tr: %d\n", II, t->forw);
	#endif
#endif
	switch (t->forw) {
	default: Uerror("bad forward move");
	case 0:	/* if without executable clauses */
		continue;
	case 1: /* generic 'goto' or 'skip' */
		IfNotBlocked
		_m = 3; goto P999;
	case 2: /* generic 'else' */
		IfNotBlocked
		if (trpt->o_pm&1) continue;
		_m = 3; goto P999;

		 /* CLAIM e6 */
	case 3: // STATE 1 - _spin_nvr.tmp:54 - [(!((!((e2==1))||((s2==2)&&(s0==2)))))] (6:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[8][1] = 1;
		if (!( !(( !((((int)now.e2)==1))||((now.s2==2)&&(now.s0==2))))))
			continue;
		/* merge: assert(!(!((!((e2==1))||((s2==2)&&(s0==2))))))(0, 2, 6) */
		reached[8][2] = 1;
		spin_assert( !( !(( !((((int)now.e2)==1))||((now.s2==2)&&(now.s0==2))))), " !( !(( !((e2==1))||((s2==2)&&(s0==2)))))", II, tt, t);
		/* merge: .(goto)(0, 7, 6) */
		reached[8][7] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 4: // STATE 10 - _spin_nvr.tmp:59 - [-end-] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported10 = 0;
			if (verbose && !reported10)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported10 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported10 = 0;
			if (verbose && !reported10)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported10 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[8][10] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* CLAIM e5 */
	case 5: // STATE 1 - _spin_nvr.tmp:43 - [((!(!((t2==1)))&&!((e2==1))))] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[7][1] = 1;
		if (!(( !( !((((int)now.t2)==1)))&& !((((int)now.e2)==1)))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 6: // STATE 8 - _spin_nvr.tmp:48 - [(!((e2==1)))] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported8 = 0;
			if (verbose && !reported8)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported8 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported8 = 0;
			if (verbose && !reported8)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported8 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[7][8] = 1;
		if (!( !((((int)now.e2)==1))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 7: // STATE 13 - _spin_nvr.tmp:50 - [-end-] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported13 = 0;
			if (verbose && !reported13)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported13 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported13 = 0;
			if (verbose && !reported13)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported13 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[7][13] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* CLAIM e4 */
	case 8: // STATE 1 - _spin_nvr.tmp:34 - [(!((!((e1==1))||((s1==1)&&(s2==1)))))] (6:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[6][1] = 1;
		if (!( !(( !((((int)now.e1)==1))||((now.s1==1)&&(now.s2==1))))))
			continue;
		/* merge: assert(!(!((!((e1==1))||((s1==1)&&(s2==1))))))(0, 2, 6) */
		reached[6][2] = 1;
		spin_assert( !( !(( !((((int)now.e1)==1))||((now.s1==1)&&(now.s2==1))))), " !( !(( !((e1==1))||((s1==1)&&(s2==1)))))", II, tt, t);
		/* merge: .(goto)(0, 7, 6) */
		reached[6][7] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 9: // STATE 10 - _spin_nvr.tmp:39 - [-end-] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported10 = 0;
			if (verbose && !reported10)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported10 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported10 = 0;
			if (verbose && !reported10)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported10 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[6][10] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* CLAIM e3 */
	case 10: // STATE 1 - _spin_nvr.tmp:23 - [((!(!((t1==1)))&&!((e1==1))))] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[5][1] = 1;
		if (!(( !( !((((int)now.t1)==1)))&& !((((int)now.e1)==1)))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 11: // STATE 8 - _spin_nvr.tmp:28 - [(!((e1==1)))] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported8 = 0;
			if (verbose && !reported8)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported8 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported8 = 0;
			if (verbose && !reported8)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported8 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[5][8] = 1;
		if (!( !((((int)now.e1)==1))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 12: // STATE 13 - _spin_nvr.tmp:30 - [-end-] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported13 = 0;
			if (verbose && !reported13)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported13 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported13 = 0;
			if (verbose && !reported13)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported13 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[5][13] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* CLAIM e2 */
	case 13: // STATE 1 - _spin_nvr.tmp:14 - [(!((!((e0==1))||((s0==0)&&(s1==0)))))] (6:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[4][1] = 1;
		if (!( !(( !((((int)now.e0)==1))||((now.s0==0)&&(now.s1==0))))))
			continue;
		/* merge: assert(!(!((!((e0==1))||((s0==0)&&(s1==0))))))(0, 2, 6) */
		reached[4][2] = 1;
		spin_assert( !( !(( !((((int)now.e0)==1))||((now.s0==0)&&(now.s1==0))))), " !( !(( !((e0==1))||((s0==0)&&(s1==0)))))", II, tt, t);
		/* merge: .(goto)(0, 7, 6) */
		reached[4][7] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 14: // STATE 10 - _spin_nvr.tmp:19 - [-end-] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported10 = 0;
			if (verbose && !reported10)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported10 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported10 = 0;
			if (verbose && !reported10)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported10 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[4][10] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* CLAIM e1 */
	case 15: // STATE 1 - _spin_nvr.tmp:3 - [((!(!((t0==1)))&&!((e0==1))))] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported1 = 0;
			if (verbose && !reported1)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported1 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[3][1] = 1;
		if (!(( !( !((((int)now.t0)==1)))&& !((((int)now.e0)==1)))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 16: // STATE 8 - _spin_nvr.tmp:8 - [(!((e0==1)))] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported8 = 0;
			if (verbose && !reported8)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported8 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported8 = 0;
			if (verbose && !reported8)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported8 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[3][8] = 1;
		if (!( !((((int)now.e0)==1))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 17: // STATE 13 - _spin_nvr.tmp:10 - [-end-] (0:0:0 - 1)
		
#if defined(VERI) && !defined(NP)
#if NCLAIMS>1
		{	static int reported13 = 0;
			if (verbose && !reported13)
			{	int nn = (int) ((Pclaim *)pptr(0))->_n;
				printf("depth %ld: Claim %s (%d), state %d (line %d)\n",
					depth, procname[spin_c_typ[nn]], nn, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported13 = 1;
				fflush(stdout);
		}	}
#else
		{	static int reported13 = 0;
			if (verbose && !reported13)
			{	printf("depth %d: Claim, state %d (line %d)\n",
					(int) depth, (int) ((Pclaim *)pptr(0))->_p, src_claim[ (int) ((Pclaim *)pptr(0))->_p ]);
				reported13 = 1;
				fflush(stdout);
		}	}
#endif
#endif
		reached[3][13] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* PROC p2 */
	case 18: // STATE 1 - ./output/mutex.pml:52 - [wi0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[2][1] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 0;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 19: // STATE 2 - ./output/mutex.pml:52 - [wi0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[2][2] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 1;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 20: // STATE 5 - ./output/mutex.pml:54 - [((wi2==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[2][5] = 1;
		if (!((((int)now.wi2)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 21: // STATE 6 - ./output/mutex.pml:55 - [wi2 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[2][6] = 1;
		(trpt+1)->bup.oval = ((int)now.wi2);
		now.wi2 = 0;
#ifdef VAR_RANGES
		logval("wi2", ((int)now.wi2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 22: // STATE 7 - ./output/mutex.pml:55 - [wi2 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[2][7] = 1;
		(trpt+1)->bup.oval = ((int)now.wi2);
		now.wi2 = 1;
#ifdef VAR_RANGES
		logval("wi2", ((int)now.wi2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 23: // STATE 10 - ./output/mutex.pml:56 - [t2 = 1] (0:0:1 - 3)
		IfNotBlocked
		reached[2][10] = 1;
		(trpt+1)->bup.oval = ((int)now.t2);
		now.t2 = 1;
#ifdef VAR_RANGES
		logval("t2", ((int)now.t2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 24: // STATE 11 - ./output/mutex.pml:57 - [t2 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[2][11] = 1;
		(trpt+1)->bup.oval = ((int)now.t2);
		now.t2 = 0;
#ifdef VAR_RANGES
		logval("t2", ((int)now.t2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 25: // STATE 12 - ./output/mutex.pml:59 - [((v0!=1))] (0:0:0 - 1)
		IfNotBlocked
		reached[2][12] = 1;
		if (!((((int)now.v0)!=1)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 26: // STATE 15 - ./output/mutex.pml:61 - [e2 = 1] (0:0:1 - 2)
		IfNotBlocked
		reached[2][15] = 1;
		(trpt+1)->bup.oval = ((int)now.e2);
		now.e2 = 1;
#ifdef VAR_RANGES
		logval("e2", ((int)now.e2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 27: // STATE 16 - ./output/mutex.pml:62 - [e2 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[2][16] = 1;
		(trpt+1)->bup.oval = ((int)now.e2);
		now.e2 = 0;
#ifdef VAR_RANGES
		logval("e2", ((int)now.e2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 28: // STATE 17 - ./output/mutex.pml:64 - [((v1==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[2][17] = 1;
		if (!((((int)now.v1)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 29: // STATE 18 - ./output/mutex.pml:65 - [v2 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[2][18] = 1;
		(trpt+1)->bup.oval = ((int)now.v2);
		now.v2 = 1;
#ifdef VAR_RANGES
		logval("v2", ((int)now.v2));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 30: // STATE 21 - ./output/mutex.pml:67 - [((s0==2))] (26:0:1 - 1)
		IfNotBlocked
		reached[2][21] = 1;
		if (!((now.s0==2)))
			continue;
		/* merge: s0 = -(1)(0, 22, 26) */
		reached[2][22] = 1;
		(trpt+1)->bup.oval = now.s0;
		now.s0 =  -(1);
#ifdef VAR_RANGES
		logval("s0", now.s0);
#endif
		;
		/* merge: .(goto)(0, 27, 26) */
		reached[2][27] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 31: // STATE 29 - ./output/mutex.pml:70 - [-end-] (0:0:0 - 3)
		IfNotBlocked
		reached[2][29] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* PROC p1 */
	case 32: // STATE 1 - ./output/mutex.pml:30 - [wi0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][1] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 0;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 33: // STATE 2 - ./output/mutex.pml:30 - [wi0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][2] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 1;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 34: // STATE 5 - ./output/mutex.pml:32 - [((wi1==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[1][5] = 1;
		if (!((((int)now.wi1)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 35: // STATE 6 - ./output/mutex.pml:33 - [wi1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][6] = 1;
		(trpt+1)->bup.oval = ((int)now.wi1);
		now.wi1 = 0;
#ifdef VAR_RANGES
		logval("wi1", ((int)now.wi1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 36: // STATE 7 - ./output/mutex.pml:33 - [wi1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][7] = 1;
		(trpt+1)->bup.oval = ((int)now.wi1);
		now.wi1 = 1;
#ifdef VAR_RANGES
		logval("wi1", ((int)now.wi1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 37: // STATE 10 - ./output/mutex.pml:34 - [t1 = 1] (0:0:1 - 3)
		IfNotBlocked
		reached[1][10] = 1;
		(trpt+1)->bup.oval = ((int)now.t1);
		now.t1 = 1;
#ifdef VAR_RANGES
		logval("t1", ((int)now.t1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 38: // STATE 11 - ./output/mutex.pml:35 - [t1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][11] = 1;
		(trpt+1)->bup.oval = ((int)now.t1);
		now.t1 = 0;
#ifdef VAR_RANGES
		logval("t1", ((int)now.t1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 39: // STATE 12 - ./output/mutex.pml:37 - [((v2!=1))] (0:0:0 - 1)
		IfNotBlocked
		reached[1][12] = 1;
		if (!((((int)now.v2)!=1)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 40: // STATE 15 - ./output/mutex.pml:39 - [e1 = 1] (0:0:1 - 2)
		IfNotBlocked
		reached[1][15] = 1;
		(trpt+1)->bup.oval = ((int)now.e1);
		now.e1 = 1;
#ifdef VAR_RANGES
		logval("e1", ((int)now.e1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 41: // STATE 16 - ./output/mutex.pml:40 - [e1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][16] = 1;
		(trpt+1)->bup.oval = ((int)now.e1);
		now.e1 = 0;
#ifdef VAR_RANGES
		logval("e1", ((int)now.e1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 42: // STATE 17 - ./output/mutex.pml:42 - [((v0==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[1][17] = 1;
		if (!((((int)now.v0)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 43: // STATE 18 - ./output/mutex.pml:43 - [v1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][18] = 1;
		(trpt+1)->bup.oval = ((int)now.v1);
		now.v1 = 1;
#ifdef VAR_RANGES
		logval("v1", ((int)now.v1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 44: // STATE 21 - ./output/mutex.pml:45 - [((s2==1))] (26:0:1 - 1)
		IfNotBlocked
		reached[1][21] = 1;
		if (!((now.s2==1)))
			continue;
		/* merge: s2 = -(1)(0, 22, 26) */
		reached[1][22] = 1;
		(trpt+1)->bup.oval = now.s2;
		now.s2 =  -(1);
#ifdef VAR_RANGES
		logval("s2", now.s2);
#endif
		;
		/* merge: .(goto)(0, 27, 26) */
		reached[1][27] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 45: // STATE 29 - ./output/mutex.pml:48 - [-end-] (0:0:0 - 3)
		IfNotBlocked
		reached[1][29] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* PROC p0 */
	case 46: // STATE 1 - ./output/mutex.pml:8 - [wi0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][1] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 0;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 47: // STATE 2 - ./output/mutex.pml:8 - [wi0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][2] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 1;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 48: // STATE 5 - ./output/mutex.pml:10 - [((wi0==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[0][5] = 1;
		if (!((((int)now.wi0)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 49: // STATE 6 - ./output/mutex.pml:11 - [wi0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][6] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 0;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 50: // STATE 7 - ./output/mutex.pml:11 - [wi0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][7] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 1;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 51: // STATE 10 - ./output/mutex.pml:12 - [t0 = 1] (0:0:1 - 3)
		IfNotBlocked
		reached[0][10] = 1;
		(trpt+1)->bup.oval = ((int)now.t0);
		now.t0 = 1;
#ifdef VAR_RANGES
		logval("t0", ((int)now.t0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 52: // STATE 11 - ./output/mutex.pml:13 - [t0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][11] = 1;
		(trpt+1)->bup.oval = ((int)now.t0);
		now.t0 = 0;
#ifdef VAR_RANGES
		logval("t0", ((int)now.t0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 53: // STATE 12 - ./output/mutex.pml:15 - [((v1!=1))] (0:0:0 - 1)
		IfNotBlocked
		reached[0][12] = 1;
		if (!((((int)now.v1)!=1)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 54: // STATE 15 - ./output/mutex.pml:17 - [e0 = 1] (0:0:1 - 2)
		IfNotBlocked
		reached[0][15] = 1;
		(trpt+1)->bup.oval = ((int)now.e0);
		now.e0 = 1;
#ifdef VAR_RANGES
		logval("e0", ((int)now.e0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 55: // STATE 16 - ./output/mutex.pml:18 - [e0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][16] = 1;
		(trpt+1)->bup.oval = ((int)now.e0);
		now.e0 = 0;
#ifdef VAR_RANGES
		logval("e0", ((int)now.e0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 56: // STATE 17 - ./output/mutex.pml:20 - [((v2==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[0][17] = 1;
		if (!((((int)now.v2)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 57: // STATE 18 - ./output/mutex.pml:21 - [v0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][18] = 1;
		(trpt+1)->bup.oval = ((int)now.v0);
		now.v0 = 1;
#ifdef VAR_RANGES
		logval("v0", ((int)now.v0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 58: // STATE 21 - ./output/mutex.pml:23 - [((s1==0))] (26:0:1 - 1)
		IfNotBlocked
		reached[0][21] = 1;
		if (!((now.s1==0)))
			continue;
		/* merge: s1 = -(1)(0, 22, 26) */
		reached[0][22] = 1;
		(trpt+1)->bup.oval = now.s1;
		now.s1 =  -(1);
#ifdef VAR_RANGES
		logval("s1", now.s1);
#endif
		;
		/* merge: .(goto)(0, 27, 26) */
		reached[0][27] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 59: // STATE 29 - ./output/mutex.pml:26 - [-end-] (0:0:0 - 3)
		IfNotBlocked
		reached[0][29] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */
	case  _T5:	/* np_ */
		if (!((!(trpt->o_pm&4) && !(trpt->tau&128))))
			continue;
		/* else fall through */
	case  _T2:	/* true */
		_m = 3; goto P999;
#undef rand
	}

