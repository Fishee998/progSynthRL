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

		 /* CLAIM e3 */
	case 3: // STATE 1 - _spin_nvr.tmp:23 - [((!(!((try1==1)))&&!((cs1==1))))] (0:0:0 - 1)
		
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
		if (!(( !( !((((int)now.try1)==1)))&& !((((int)now.cs1)==1)))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 4: // STATE 8 - _spin_nvr.tmp:28 - [(!((cs1==1)))] (0:0:0 - 1)
		
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
		reached[4][8] = 1;
		if (!( !((((int)now.cs1)==1))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 5: // STATE 13 - _spin_nvr.tmp:30 - [-end-] (0:0:0 - 1)
		
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
		reached[4][13] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* CLAIM e2 */
	case 6: // STATE 1 - _spin_nvr.tmp:12 - [((!(!((try0==1)))&&!((cs0==1))))] (0:0:0 - 1)
		
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
		if (!(( !( !((((int)now.try0)==1)))&& !((((int)now.cs0)==1)))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 7: // STATE 8 - _spin_nvr.tmp:17 - [(!((cs0==1)))] (0:0:0 - 1)
		
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
		if (!( !((((int)now.cs0)==1))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 8: // STATE 13 - _spin_nvr.tmp:19 - [-end-] (0:0:0 - 1)
		
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

		 /* CLAIM e1 */
	case 9: // STATE 1 - _spin_nvr.tmp:3 - [(!(((cs0+cs1)<=1)))] (6:0:0 - 1)
		
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
		reached[2][1] = 1;
		if (!( !(((((int)now.cs0)+((int)now.cs1))<=1))))
			continue;
		/* merge: assert(!(!(((cs0+cs1)<=1))))(0, 2, 6) */
		reached[2][2] = 1;
		spin_assert( !( !(((((int)now.cs0)+((int)now.cs1))<=1))), " !( !(((cs0+cs1)<=1)))", II, tt, t);
		/* merge: .(goto)(0, 7, 6) */
		reached[2][7] = 1;
		;
		_m = 3; goto P999; /* 2 */
	case 10: // STATE 10 - _spin_nvr.tmp:8 - [-end-] (0:0:0 - 1)
		
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
		reached[2][10] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* PROC q */
	case 11: // STATE 1 - ./output/mutex.pml:34 - [wi1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][1] = 1;
		(trpt+1)->bup.oval = ((int)now.wi1);
		now.wi1 = 0;
#ifdef VAR_RANGES
		logval("wi1", ((int)now.wi1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 12: // STATE 2 - ./output/mutex.pml:34 - [wi1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][2] = 1;
		(trpt+1)->bup.oval = ((int)now.wi1);
		now.wi1 = 1;
#ifdef VAR_RANGES
		logval("wi1", ((int)now.wi1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 13: // STATE 5 - ./output/mutex.pml:36 - [((wi1==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[1][5] = 1;
		if (!((((int)now.wi1)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 14: // STATE 6 - ./output/mutex.pml:37 - [try1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][6] = 1;
		(trpt+1)->bup.oval = ((int)now.try1);
		now.try1 = 1;
#ifdef VAR_RANGES
		logval("try1", ((int)now.try1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 15: // STATE 7 - ./output/mutex.pml:38 - [wi1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][7] = 1;
		(trpt+1)->bup.oval = ((int)now.wi1);
		now.wi1 = 0;
#ifdef VAR_RANGES
		logval("wi1", ((int)now.wi1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 16: // STATE 8 - ./output/mutex.pml:38 - [wi1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][8] = 1;
		(trpt+1)->bup.oval = ((int)now.wi1);
		now.wi1 = 1;
#ifdef VAR_RANGES
		logval("wi1", ((int)now.wi1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 17: // STATE 11 - ./output/mutex.pml:39 - [v0 = 0] (0:0:1 - 3)
		IfNotBlocked
		reached[1][11] = 1;
		(trpt+1)->bup.oval = ((int)now.v0);
		now.v0 = 0;
#ifdef VAR_RANGES
		logval("v0", ((int)now.v0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 18: // STATE 12 - ./output/mutex.pml:40 - [turn = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][12] = 1;
		(trpt+1)->bup.oval = ((int)turn);
		turn = 0;
#ifdef VAR_RANGES
		logval("turn", ((int)turn));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 19: // STATE 13 - ./output/mutex.pml:41 - [v0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][13] = 1;
		(trpt+1)->bup.oval = ((int)now.v0);
		now.v0 = 1;
#ifdef VAR_RANGES
		logval("v0", ((int)now.v0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 20: // STATE 14 - ./output/mutex.pml:42 - [turn = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][14] = 1;
		(trpt+1)->bup.oval = ((int)turn);
		turn = 0;
#ifdef VAR_RANGES
		logval("turn", ((int)turn));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 21: // STATE 15 - ./output/mutex.pml:44 - [(((v1==1)&&(v0!=0)))] (0:0:0 - 1)
		IfNotBlocked
		reached[1][15] = 1;
		if (!(((((int)now.v1)==1)&&(((int)now.v0)!=0))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 22: // STATE 16 - ./output/mutex.pml:46 - [((v1!=0))] (0:0:0 - 1)
		IfNotBlocked
		reached[1][16] = 1;
		if (!((((int)now.v1)!=0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 23: // STATE 17 - ./output/mutex.pml:47 - [v1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][17] = 1;
		(trpt+1)->bup.oval = ((int)now.v1);
		now.v1 = 0;
#ifdef VAR_RANGES
		logval("v1", ((int)now.v1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 24: // STATE 18 - ./output/mutex.pml:48 - [v1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[1][18] = 1;
		(trpt+1)->bup.oval = ((int)now.v1);
		now.v1 = 1;
#ifdef VAR_RANGES
		logval("v1", ((int)now.v1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 25: // STATE 26 - ./output/mutex.pml:52 - [cs1 = (cs1+1)] (0:0:1 - 3)
		IfNotBlocked
		reached[1][26] = 1;
		(trpt+1)->bup.oval = ((int)now.cs1);
		now.cs1 = (((int)now.cs1)+1);
#ifdef VAR_RANGES
		logval("cs1", ((int)now.cs1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 26: // STATE 27 - ./output/mutex.pml:53 - [try1 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[1][27] = 1;
		(trpt+1)->bup.oval = ((int)now.try1);
		now.try1 = 0;
#ifdef VAR_RANGES
		logval("try1", ((int)now.try1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 27: // STATE 28 - ./output/mutex.pml:54 - [cs1 = (cs1-1)] (0:0:1 - 1)
		IfNotBlocked
		reached[1][28] = 1;
		(trpt+1)->bup.oval = ((int)now.cs1);
		now.cs1 = (((int)now.cs1)-1);
#ifdef VAR_RANGES
		logval("cs1", ((int)now.cs1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 28: // STATE 34 - ./output/mutex.pml:57 - [-end-] (0:0:0 - 3)
		IfNotBlocked
		reached[1][34] = 1;
		if (!delproc(1, II)) continue;
		_m = 3; goto P999; /* 0 */

		 /* PROC p */
	case 29: // STATE 1 - ./output/mutex.pml:7 - [wi0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][1] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 0;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 30: // STATE 2 - ./output/mutex.pml:7 - [wi0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][2] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 1;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 31: // STATE 5 - ./output/mutex.pml:9 - [((wi0==0))] (0:0:0 - 1)
		IfNotBlocked
		reached[0][5] = 1;
		if (!((((int)now.wi0)==0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 32: // STATE 6 - ./output/mutex.pml:10 - [try0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][6] = 1;
		(trpt+1)->bup.oval = ((int)now.try0);
		now.try0 = 1;
#ifdef VAR_RANGES
		logval("try0", ((int)now.try0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 33: // STATE 7 - ./output/mutex.pml:11 - [wi0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][7] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 0;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 34: // STATE 8 - ./output/mutex.pml:11 - [wi0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][8] = 1;
		(trpt+1)->bup.oval = ((int)now.wi0);
		now.wi0 = 1;
#ifdef VAR_RANGES
		logval("wi0", ((int)now.wi0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 35: // STATE 11 - ./output/mutex.pml:12 - [v1 = 1] (0:0:1 - 3)
		IfNotBlocked
		reached[0][11] = 1;
		(trpt+1)->bup.oval = ((int)now.v1);
		now.v1 = 1;
#ifdef VAR_RANGES
		logval("v1", ((int)now.v1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 36: // STATE 12 - ./output/mutex.pml:13 - [turn = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][12] = 1;
		(trpt+1)->bup.oval = ((int)turn);
		turn = 0;
#ifdef VAR_RANGES
		logval("turn", ((int)turn));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 37: // STATE 13 - ./output/mutex.pml:14 - [v1 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][13] = 1;
		(trpt+1)->bup.oval = ((int)now.v1);
		now.v1 = 1;
#ifdef VAR_RANGES
		logval("v1", ((int)now.v1));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 38: // STATE 14 - ./output/mutex.pml:15 - [turn = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][14] = 1;
		(trpt+1)->bup.oval = ((int)turn);
		turn = 0;
#ifdef VAR_RANGES
		logval("turn", ((int)turn));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 39: // STATE 15 - ./output/mutex.pml:17 - [(((v1==1)&&(v0!=0)))] (0:0:0 - 1)
		IfNotBlocked
		reached[0][15] = 1;
		if (!(((((int)now.v1)==1)&&(((int)now.v0)!=0))))
			continue;
		_m = 3; goto P999; /* 0 */
	case 40: // STATE 16 - ./output/mutex.pml:19 - [((v1!=0))] (0:0:0 - 1)
		IfNotBlocked
		reached[0][16] = 1;
		if (!((((int)now.v1)!=0)))
			continue;
		_m = 3; goto P999; /* 0 */
	case 41: // STATE 17 - ./output/mutex.pml:20 - [v0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][17] = 1;
		(trpt+1)->bup.oval = ((int)now.v0);
		now.v0 = 0;
#ifdef VAR_RANGES
		logval("v0", ((int)now.v0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 42: // STATE 18 - ./output/mutex.pml:21 - [v0 = 1] (0:0:1 - 1)
		IfNotBlocked
		reached[0][18] = 1;
		(trpt+1)->bup.oval = ((int)now.v0);
		now.v0 = 1;
#ifdef VAR_RANGES
		logval("v0", ((int)now.v0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 43: // STATE 26 - ./output/mutex.pml:25 - [cs0 = (cs0+1)] (0:0:1 - 3)
		IfNotBlocked
		reached[0][26] = 1;
		(trpt+1)->bup.oval = ((int)now.cs0);
		now.cs0 = (((int)now.cs0)+1);
#ifdef VAR_RANGES
		logval("cs0", ((int)now.cs0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 44: // STATE 27 - ./output/mutex.pml:26 - [try0 = 0] (0:0:1 - 1)
		IfNotBlocked
		reached[0][27] = 1;
		(trpt+1)->bup.oval = ((int)now.try0);
		now.try0 = 0;
#ifdef VAR_RANGES
		logval("try0", ((int)now.try0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 45: // STATE 28 - ./output/mutex.pml:27 - [cs0 = (cs0-1)] (0:0:1 - 1)
		IfNotBlocked
		reached[0][28] = 1;
		(trpt+1)->bup.oval = ((int)now.cs0);
		now.cs0 = (((int)now.cs0)-1);
#ifdef VAR_RANGES
		logval("cs0", ((int)now.cs0));
#endif
		;
		_m = 3; goto P999; /* 0 */
	case 46: // STATE 34 - ./output/mutex.pml:30 - [-end-] (0:0:0 - 3)
		IfNotBlocked
		reached[0][34] = 1;
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

