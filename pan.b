	switch (t->back) {
	default: Uerror("bad return move");
	case  0: goto R999; /* nothing to undo */

		 /* CLAIM e3 */
;
		;
		;
		;
		
	case 5: // STATE 13
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e2 */
;
		;
		;
		;
		
	case 8: // STATE 13
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* CLAIM e1 */
;
		
	case 9: // STATE 1
		goto R999;

	case 10: // STATE 10
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* PROC q */

	case 11: // STATE 1
		;
		now.wi1 = trpt->bup.oval;
		;
		goto R999;

	case 12: // STATE 2
		;
		now.wi1 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 14: // STATE 6
		;
		now.try1 = trpt->bup.oval;
		;
		goto R999;

	case 15: // STATE 7
		;
		now.wi1 = trpt->bup.oval;
		;
		goto R999;

	case 16: // STATE 8
		;
		now.wi1 = trpt->bup.oval;
		;
		goto R999;

	case 17: // STATE 11
		;
		now.v0 = trpt->bup.oval;
		;
		goto R999;

	case 18: // STATE 12
		;
		turn = trpt->bup.oval;
		;
		goto R999;

	case 19: // STATE 13
		;
		now.v0 = trpt->bup.oval;
		;
		goto R999;

	case 20: // STATE 14
		;
		turn = trpt->bup.oval;
		;
		goto R999;
;
		;
		;
		;
		
	case 23: // STATE 17
		;
		now.v1 = trpt->bup.oval;
		;
		goto R999;

	case 24: // STATE 18
		;
		now.v1 = trpt->bup.oval;
		;
		goto R999;

	case 25: // STATE 26
		;
		now.cs1 = trpt->bup.oval;
		;
		goto R999;

	case 26: // STATE 27
		;
		now.try1 = trpt->bup.oval;
		;
		goto R999;

	case 27: // STATE 28
		;
		now.cs1 = trpt->bup.oval;
		;
		goto R999;

	case 28: // STATE 34
		;
		p_restor(II);
		;
		;
		goto R999;

		 /* PROC p */

	case 29: // STATE 1
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 30: // STATE 2
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;
;
		;
		
	case 32: // STATE 6
		;
		now.try0 = trpt->bup.oval;
		;
		goto R999;

	case 33: // STATE 7
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 34: // STATE 8
		;
		now.wi0 = trpt->bup.oval;
		;
		goto R999;

	case 35: // STATE 11
		;
		now.v1 = trpt->bup.oval;
		;
		goto R999;

	case 36: // STATE 12
		;
		turn = trpt->bup.oval;
		;
		goto R999;

	case 37: // STATE 13
		;
		now.v1 = trpt->bup.oval;
		;
		goto R999;

	case 38: // STATE 14
		;
		turn = trpt->bup.oval;
		;
		goto R999;
;
		;
		;
		;
		
	case 41: // STATE 17
		;
		now.v0 = trpt->bup.oval;
		;
		goto R999;

	case 42: // STATE 18
		;
		now.v0 = trpt->bup.oval;
		;
		goto R999;

	case 43: // STATE 26
		;
		now.cs0 = trpt->bup.oval;
		;
		goto R999;

	case 44: // STATE 27
		;
		now.try0 = trpt->bup.oval;
		;
		goto R999;

	case 45: // STATE 28
		;
		now.cs0 = trpt->bup.oval;
		;
		goto R999;

	case 46: // STATE 34
		;
		p_restor(II);
		;
		;
		goto R999;
	}

