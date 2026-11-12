/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __LINUX_TEXT_KSM_H
#define __LINUX_TEXT_KSM_H
/*
 * text_ksm - merging of file-backed read-only executable pages.
 *
 * A separate subsystem from the traditional (anonymous) KSM in ksm.c. See
 * mm/text_ksm.c for the full design. A process opts in per-mm via
 * prctl(PR_SET_TEXT_MERGE); the ktextksmd thread then merges identical
 * file pages (e.g. the same shared library mapped from different paths)
 * across the opted-in mms.
 */

#include <linux/mm.h>

#ifdef CONFIG_TEXT_KSM
int text_ksm_enable(struct mm_struct *mm);
int text_ksm_disable(struct mm_struct *mm);

int __text_ksm_enter(struct mm_struct *mm);
void __text_ksm_exit(struct mm_struct *mm);

static inline void text_ksm_fork(struct mm_struct *mm, struct mm_struct *oldmm)
{
	/* Propagate the opt-in on fork, best effort. */
	if (mm_flags_test(MMF_TEXT_MERGEABLE, oldmm))
		__text_ksm_enter(mm);
}

static inline void text_ksm_exit(struct mm_struct *mm)
{
	if (mm_flags_test(MMF_TEXT_MERGEABLE, mm))
		__text_ksm_exit(mm);
}

#else  /* !CONFIG_TEXT_KSM */

static inline int text_ksm_enable(struct mm_struct *mm)
{
	return 0;
}

static inline int text_ksm_disable(struct mm_struct *mm)
{
	return 0;
}

static inline void text_ksm_fork(struct mm_struct *mm, struct mm_struct *oldmm)
{
}

static inline void text_ksm_exit(struct mm_struct *mm)
{
}

#endif /* !CONFIG_TEXT_KSM */

#endif /* __LINUX_TEXT_KSM_H */
