	switch (t->back) {
	default: Uerror("bad return move");
	case  0: goto R999; /* nothing to undo */

		 /* CLAIM e6 */
;
		
	case 3: // STATE 1
		goto R999;

	case 4: // STATE 10
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e5 */
;
		;
		;
		;
		
	case 7: // STATE 13
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e4 */
;
		
	case 8: // STATE 1
		goto R999;

	case 9: // STATE 10
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e3 */
;
		;
		;
		;
		
	case 12: // STATE 13
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e2 */
;
		
	case 13: // STATE 1
		goto R999;

	case 14: // STATE 10
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e1 */
;
		;
		;
		;
		
	case 17: // STATE 13
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* PROC p2 */

	case 18: // STATE 1
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 19: // STATE 2
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 21: // STATE 6
		;
		now.wi2 = trpt->bup.oval;
		;
		goto R999;

	case 22: // STATE 7
		;
		now.wi2 = trpt->bup.oval;
		;
		goto R999;

	case 23: // STATE 10
		;
		now.t2 = trpt->bup.oval;
		;
		goto R999;

	case 24: // STATE 11
		;
		now.t2 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 26: // STATE 15
		;
		now.e2 = trpt->bup.oval;
		;
		goto R999;

	case 27: // STATE 16
		;
		now.e2 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 29: // STATE 18
		;
		now.v2 = trpt->bup.oval;
		;
		goto R999;

	case 30: // STATE 22
		;
		now.s0 = trpt->bup.oval;
		;
		goto R999;

	case 31: // STATE 29
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* PROC p1 */

	case 32: // STATE 1
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 33: // STATE 2
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 35: // STATE 6
		;
		now.wi1 = trpt->bup.oval;
		;
		goto R999;

	case 36: // STATE 7
		;
		now.wi1 = trpt->bup.oval;
		;
		goto R999;

	case 37: // STATE 10
		;
		now.t1 = trpt->bup.oval;
		;
		goto R999;

	case 38: // STATE 11
		;
		now.t1 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 40: // STATE 15
		;
		now.e1 = trpt->bup.oval;
		;
		goto R999;

	case 41: // STATE 16
		;
		now.e1 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 43: // STATE 18
		;
		now.v1 = trpt->bup.oval;
		;
		goto R999;

	case 44: // STATE 22
		;
		now.s2 = trpt->bup.oval;
		;
		goto R999;

	case 45: // STATE 29
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* PROC p0 */

	case 46: // STATE 1
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 47: // STATE 2
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 49: // STATE 6
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 50: // STATE 7
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 51: // STATE 10
		;
		now.t0 = trpt->bup.oval;
		;
		goto R999;

	case 52: // STATE 11
		;
		now.t0 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 54: // STATE 15
		;
		now.e0 = trpt->bup.oval;
		;
		goto R999;

	case 55: // STATE 16
		;
		now.e0 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 57: // STATE 18
		;
		now.v0 = trpt->bup.oval;
		;
		goto R999;

	case 58: // STATE 22
		;
		now.s1 = trpt->bup.oval;
		;
		goto R999;

	case 59: // STATE 29
		;
		p_restor(II);
		;
		;
		goto R999;
	}

