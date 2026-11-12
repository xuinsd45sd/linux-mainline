// SPDX-License-Identifier: GPL-2.0-only
/*
 * text_ksm - detection of file-backed read-only executable pages.
 *
 * This is a separate subsystem from the traditional (anonymous) KSM in
 * ksm.c. It targets *file* pages (e.g. the same shared library mapped from
 * different paths / inodes) rather than anonymous pages. The scan logic is
 * fundamentally different: there is no rmap_item, no unstable tree, and no
 * per-VMA opt-in. Instead:
 *
 *   - a process opts in per-mm via prctl(PR_SET_TEXT_MERGE), which adds its
 *     mm to text_ksm's own mm list (independent of KSM's ksm_mm_head);
 *   - the ktextksmd thread walks the page cache of every eligible file VMA
 *     (file-backed, PROT_EXEC, currently read-only, inode passing the
 *     read-only predicate) of the opted-in mms, content-matching the resident
 *     single-page folios against a content-keyed stable tree.
 *
 * This is detection and content matching only: no pages are merged yet. The
 * counters let us cross-check the in-kernel view against the userspace
 * ksm-survey tool.
 *
 * The sysfs interface lives under /sys/kernel/mm/text_ksm/ (independent of
 * /sys/kernel/mm/ksm/), and the scan thread is ktextksmd (independent of
 * ksmd), with its own run / pages_to_scan / sleep_millisecs knobs.
 */

#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/rwsem.h>
#include <linux/pagemap.h>
#include <linux/rmap.h>
#include <linux/spinlock.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/slab.h>
#include <linux/rbtree.h>
#include <linux/hashtable.h>
#include <linux/freezer.h>
#include <linux/delay.h>
#include <linux/sysfs.h>
#include <linux/kobject.h>

#include <linux/text_ksm.h>

#include "internal.h"
#include "mm_slot.h"

/*
 * struct text_ksm_mm_slot - per-mm slot for the text_ksm mm list
 * @slot: hash lookup from mm to mm_slot
 * @scan_done: set while this mm's file VMAs have been walked for the current
 *             scan; reset at the start of each scan. Protected by
 *             text_ksm_mmlist_lock. Lets the scan iterate text_ksm_mm_head
 *             without depending on the (possibly freed) slot of the mm it
 *             last visited.
 */
struct text_ksm_mm_slot {
	struct mm_slot slot;
	bool scan_done;
};

/*
 * struct text_ksm_scan - cursor for the text_ksm scan.
 *
 * The scan is stateful and budget-limited: each call to text_ksm_do_scan()
 * scans at most @scan_npages page-cache indices and resumes where it left
 * off on the next call, so ktextksmd's CPU usage stays bounded by
 * text_ksm_pages_to_scan.
 *
 * mm is mmgrab'd by the scan itself (independent of the mm_slot's
 * reference, which __text_ksm_exit may drop across cycles), so the mm stays
 * alive while the cursor is parked on it. inode is one of the
 * text_ksm_seen_inodes entries (igrab'd for the whole scan); it is NULL
 * between mappings.
 */
struct text_ksm_scan {
	struct mm_struct *mm;		/* current mm (mmgrab'd), or NULL */
	unsigned long address;		/* next VMA address to scan in @mm */
	struct inode *inode;		/* mapping being walked, or NULL */
	pgoff_t index;			/* next page-cache index to scan */
	pgoff_t max_index;		/* end of @inode's page cache */
};
static struct text_ksm_scan text_ksm_scan;

/*
 * Per-scan set of inodes already walked, so a file mapped by several mms
 * (e.g. libc in many processes) has its page cache counted only once per
 * scan. Without this, the second walk would count the same page-cache pages
 * again and inflate the totals. Each inode is igrab'd for the duration of
 * the scan; the list is emptied (and the inodes iput) when a full scan
 * completes.
 */
struct text_ksm_seen_inode {
	struct list_head list;
	struct inode *inode;		/* igrab'd */
};
static LIST_HEAD(text_ksm_seen_inodes);

/*
 * struct text_ksm_stable_node - node of the content-keyed stable tree.
 *
 * Independent from the anonymous KSM stable tree (type isolation: file pages
 * only merge with file pages). Like the anon stable tree it is content-keyed:
 * nodes are ordered by page content (memcmp_pages), so a search finds a page
 * with identical content.
 *
 * Each node holds a reference on its survivor's folio (via the page pointer),
 * so the survivor is always valid for comparison — no keyhole check is needed
 * in this measurement pass. (A later patch switches to a ref-free,
 * identity-based keyhole so that a merged page can remain evictable.)
 */
struct text_ksm_stable_node {
	struct rb_node node;
	struct page *page;		/* survivor page; ref on page_folio(page) */
};
static struct rb_root root_text_ksm_stable_tree = RB_ROOT;

#define TEXT_KSM_MM_SLOTS_HASH_BITS 10
static DEFINE_HASHTABLE(text_ksm_mm_slots_hash, TEXT_KSM_MM_SLOTS_HASH_BITS);

static struct text_ksm_mm_slot text_ksm_mm_head = {
	.slot.mm_node = LIST_HEAD_INIT(text_ksm_mm_head.slot.mm_node),
};
static struct kmem_cache *text_ksm_mm_slot_cache;

static DEFINE_SPINLOCK(text_ksm_mmlist_lock);

/*
 * text_ksm_test_exit() is used throughout to make this test for exit: in some
 * places for correctness, in some places just to avoid unnecessary work.
 */
static inline bool text_ksm_test_exit(struct mm_struct *mm)
{
	return atomic_read(&mm->mm_users) == 0;
}

/* Default number of pages to scan per batch */
#define TEXT_KSM_DEFAULT_PAGES_TO_SCAN 100

/* Whether ktextksmd should be scanning */
static bool text_ksm_run __read_mostly;

/* Number of pages ktextksmd should scan in one batch */
static unsigned int text_ksm_pages_to_scan = TEXT_KSM_DEFAULT_PAGES_TO_SCAN;

/* Milliseconds ktextksmd should sleep between batches */
static unsigned int text_ksm_sleep_millisecs = 20;

/*
 * True while a text_ksm scan is in progress (the seen-inode list is
 * populated). Set when the scan starts, cleared when a full scan completes
 * or the scan is disabled.
 */
static bool text_ksm_scan_active;

/*
 * Scan counters, exposed via sysfs. They hold the result of the last
 * *completed* scan: the scan accumulates into the private text_ksm_*_acc
 * variables and publishes them to these only when a full scan finishes
 * (text_ksm_scan_finish). This keeps them stable — a reader never sees a
 * half-finished scan, even though the scan restarts every cycle.
 *
 * pages_total:    upper bound — total pages in eligible file-backed
 *                 PROT_EXEC read-only VMAs (whole VMA ranges, incl.
 *                 non-resident).
 * pages_resident: of the eligible mappings, pages actually resident in the
 *                 page cache (single-page folios, THP skipped).
 * pages_mergeable: of the resident pages, those with an identical
 *                 counterpart already seen this scan (i.e. actually
 *                 mergeable by content).
 */
static unsigned long text_ksm_pages_total;
static unsigned long text_ksm_pages_resident;
static unsigned long text_ksm_pages_mergeable;

/* Private accumulators for the in-progress scan (see above). */
static unsigned long text_ksm_total_acc;
static unsigned long text_ksm_resident_acc;
static unsigned long text_ksm_mergeable_acc;

static DECLARE_WAIT_QUEUE_HEAD(text_ksm_thread_wait);
static DECLARE_WAIT_QUEUE_HEAD(text_ksm_iter_wait);
static DEFINE_MUTEX(text_ksm_thread_mutex);

/*
 * text_ksm_stable_tree_search - search the stable tree for a page with
 * identical content to @page.
 *
 * Content-keyed walk (memcmp_pages), like the anonymous stable tree. Returns
 * the matching node if found, NULL otherwise. The node holds a ref on its
 * survivor's folio, so it is always valid for comparison — no keyhole check
 * is needed in this measurement pass.
 */
static struct text_ksm_stable_node *text_ksm_stable_tree_search(struct page *page)
{
	struct rb_node **new = &root_text_ksm_stable_tree.rb_node;
	struct rb_node *parent = NULL;

	while (*new) {
		struct text_ksm_stable_node *node;
		int ret;

		cond_resched();
		node = rb_entry(*new, struct text_ksm_stable_node, node);
		ret = memcmp_pages(page, node->page);

		parent = *new;
		if (ret < 0)
			new = &parent->rb_left;
		else if (ret > 0)
			new = &parent->rb_right;
		else
			return node;
	}
	return NULL;
}

/*
 * text_ksm_stable_tree_insert - insert a node for @page into the stable tree,
 * recording it as the survivor.
 *
 * Returns the new node on success, NULL if an identical page is already in
 * the tree or on allocation failure. The node takes a reference on the
 * survivor's folio so it stays alive for the duration of the scan.
 */
static struct text_ksm_stable_node *text_ksm_stable_tree_insert(struct page *page)
{
	struct folio *folio = page_folio(page);
	struct text_ksm_stable_node *node;
	struct rb_node **new = &root_text_ksm_stable_tree.rb_node;
	struct rb_node *parent = NULL;

	node = kmalloc(sizeof(*node), GFP_KERNEL);
	if (!node)
		return NULL;

	while (*new) {
		struct text_ksm_stable_node *t;
		int ret;

		cond_resched();
		t = rb_entry(*new, struct text_ksm_stable_node, node);
		ret = memcmp_pages(page, t->page);

		parent = *new;
		if (ret < 0)
			new = &parent->rb_left;
		else if (ret > 0)
			new = &parent->rb_right;
		else {
			/* identical content already in the tree */
			kfree(node);
			return NULL;
		}
	}

	/*
	 * Record the survivor page and take a reference on its folio so it
	 * stays alive for the duration of the scan.
	 */
	node->page = page;
	folio_get(folio);
	rb_link_node(&node->node, parent, new);
	rb_insert_color(&node->node, &root_text_ksm_stable_tree);
	return node;
}

/*
 * text_ksm_stable_tree_free - free all nodes in the stable tree.
 *
 * Called at the end of each scan pass, so the nodes allocated by the pass are
 * not leaked. Each node holds a reference on its survivor's folio, so drop it
 * here.
 */
static void text_ksm_stable_tree_free(void)
{
	struct rb_node *node = rb_first(&root_text_ksm_stable_tree);

	while (node) {
		struct text_ksm_stable_node *sn =
			rb_entry(node, struct text_ksm_stable_node, node);

		node = rb_next(node);
		rb_erase(&sn->node, &root_text_ksm_stable_tree);
		folio_put(page_folio(sn->page));
		kfree(sn);
	}
}

/*
 * text_ksm_scan_inode_seen - has @inode's page cache already been walked this
 * scan?
 */
static bool text_ksm_scan_inode_seen(struct inode *inode)
{
	struct text_ksm_seen_inode *si;

	list_for_each_entry(si, &text_ksm_seen_inodes, list)
		if (si->inode == inode)
			return true;
	return false;
}

/*
 * text_ksm_scan_inode_mark_seen - record @inode as walked for this scan,
 * igrab'ing it so it (and its page cache) stay alive while the cursor is
 * parked on it. Returns false on allocation failure (the caller then skips
 * the inode).
 */
static bool text_ksm_scan_inode_mark_seen(struct inode *inode)
{
	struct text_ksm_seen_inode *si;

	si = kmalloc(sizeof(*si), GFP_KERNEL);
	if (!si)
		return false;
	igrab(inode);
	si->inode = inode;
	INIT_LIST_HEAD(&si->list);
	list_add(&si->list, &text_ksm_seen_inodes);
	return true;
}

/*
 * text_ksm_scan_next_mm - pick the next mm from text_ksm_mm_head that has not
 * yet been scanned, mmgrab it into the cursor, and return true. Returns false
 * when every mm has been scanned (the scan is complete).
 *
 * The mm is mmgrab'd by the scan itself (independent of the mm_slot's
 * reference, which __text_ksm_exit may drop across cycles), so it stays alive
 * while the cursor is parked on it.
 */
static bool text_ksm_scan_next_mm(void)
{
	struct text_ksm_mm_slot *mm_slot;
	struct mm_slot *slot;

	spin_lock(&text_ksm_mmlist_lock);
	list_for_each_entry(slot, &text_ksm_mm_head.slot.mm_node, mm_node) {
		mm_slot = mm_slot_entry(slot, struct text_ksm_mm_slot, slot);
		if (!mm_slot->scan_done) {
			mmgrab(slot->mm);
			text_ksm_scan.mm = slot->mm;
			text_ksm_scan.address = 0;
			spin_unlock(&text_ksm_mmlist_lock);
			return true;
		}
	}
	spin_unlock(&text_ksm_mmlist_lock);
	return false;
}

/*
 * text_ksm_scan_done_mm - mark @mm's slot as scanned and drop the cursor's
 * reference to it.
 */
static void text_ksm_scan_done_mm(struct mm_struct *mm)
{
	struct mm_slot *slot;
	struct text_ksm_mm_slot *mm_slot;

	spin_lock(&text_ksm_mmlist_lock);
	slot = mm_slot_lookup(text_ksm_mm_slots_hash, mm);
	if (slot) {
		mm_slot = mm_slot_entry(slot, struct text_ksm_mm_slot, slot);
		mm_slot->scan_done = true;
	}
	spin_unlock(&text_ksm_mmlist_lock);

	mmdrop(mm);
	text_ksm_scan.mm = NULL;
}

/*
 * text_ksm_scan_find_next_mapping - advance the cursor to the next eligible
 * file mapping (a PROT_EXEC, read-only file VMA whose inode passes the
 * read-only predicate and has not been walked this scan), setting up
 * text_ksm_scan.inode/index/max_index. Returns true if found, false when
 * there are no more mappings (the scan is complete).
 *
 * The per-mm opt-in (prctl(PR_SET_TEXT_MERGE)) gates which mms are scanned;
 * the VMA predicate below gates which VMAs within those mms are eligible.
 * There is no per-VMA opt-in flag: an opted-in process authorizes merging of
 * all its read-only executable file pages.
 */
static bool text_ksm_scan_find_next_mapping(void)
{
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	struct vma_iterator vmi;
	struct inode *inode;

	text_ksm_scan.inode = NULL;

	while (1) {
		mm = text_ksm_scan.mm;
		if (!mm) {
			if (!text_ksm_scan_next_mm())
				return false;
			mm = text_ksm_scan.mm;
		}

		mmap_read_lock(mm);
		if (text_ksm_test_exit(mm)) {
			mmap_read_unlock(mm);
			goto next_mm;
		}
		vma_iter_init(&vmi, mm, text_ksm_scan.address);
		for_each_vma(vmi, vma) {
			/*
			 * file-backed, executable, and currently read-only.
			 *
			 * VM_WRITE (not VM_MAYWRITE) is the right test: a
			 * typical .so is mapped MAP_PRIVATE PROT_READ|PROT_EXEC,
			 * which carries VM_MAYWRITE (it *may* be made writable)
			 * but not VM_WRITE (it is not writable now).
			 */
			if (!(vma->vm_flags & VM_EXEC))
				continue;
			if (vma->vm_flags & VM_WRITE)
				continue;
			if (!vma->vm_file)
				continue;

			inode = file_inode(vma->vm_file);

			/* read-only eligibility predicate */
			if (!S_ISREG(inode->i_mode))
				continue;
			if (!(inode->i_mode & S_IXUGO))
				continue;
			if (atomic_read(&inode->i_mapping->i_mmap_writable))
				continue;
			/*
			 * i_writecount == 0: no process has the file open for
			 * writing. (A later patch relaxes this to <= 0 to also
			 * cover the execve deny_write_access case, where the
			 * kernel has locked the file read-only.)
			 */
			if (atomic_read(&inode->i_writecount))
				continue;

			/* upper bound: pages in this eligible VMA */
			text_ksm_total_acc +=
				(vma->vm_end - vma->vm_start) >> PAGE_SHIFT;

			/*
			 * dedup: a file mapped by several mms (e.g. libc in
			 * many processes) shares one page cache; walk it once.
			 */
			if (text_ksm_scan_inode_seen(inode)) {
				text_ksm_scan.address = vma->vm_end;
				continue;
			}
			if (!text_ksm_scan_inode_mark_seen(inode)) {
				text_ksm_scan.address = vma->vm_end;
				continue;
			}

			text_ksm_scan.inode = inode;
			text_ksm_scan.index = 0;
			text_ksm_scan.max_index =
				i_size_read(inode) >> PAGE_SHIFT;
			text_ksm_scan.address = vma->vm_end;
			mmap_read_unlock(mm);
			return true;
		}
		mmap_read_unlock(mm);
next_mm:
		text_ksm_scan_done_mm(mm);
		text_ksm_scan.address = 0;
	}
}

/*
 * text_ksm_scan_next - advance the scan by one page-cache index and count it
 * if it is a resident single-page folio. Returns true if an index was
 * examined (the scan may continue), false when the scan is complete (every mm
 * walked).
 */
static bool text_ksm_scan_next(void)
{
	struct address_space *mapping;
	void *entry_folio;
	struct folio *folio;

	if (!text_ksm_scan.inode || text_ksm_scan.index > text_ksm_scan.max_index) {
		if (!text_ksm_scan_find_next_mapping())
			return false;
	}

	mapping = text_ksm_scan.inode->i_mapping;
	entry_folio = filemap_get_entry(mapping, text_ksm_scan.index);
	text_ksm_scan.index++;

	if (entry_folio && !xa_is_value(entry_folio)) {
		folio = (struct folio *)entry_folio;
		/*
		 * text_ksm merges single pages only; large folios (THP) are
		 * skipped. Shared libraries don't use THP, so this is a rare
		 * skip in practice.
		 */
		if (!folio_test_large(folio)) {
			text_ksm_resident_acc++;
			/*
			 * Content-match: if an identical page is already in
			 * the stable tree this one is mergeable; otherwise
			 * record it as a new survivor.
			 */
			if (text_ksm_stable_tree_search(&folio->page))
				text_ksm_mergeable_acc++;
			else
				text_ksm_stable_tree_insert(&folio->page);
		}
		folio_put(folio);
	}
	return true;
}

/*
 * text_ksm_scan_start - begin a new scan: reset the per-mm done flags, the
 * counters, and the cursor.
 */
static void text_ksm_scan_start(void)
{
	struct text_ksm_mm_slot *mm_slot;
	struct mm_slot *slot;

	spin_lock(&text_ksm_mmlist_lock);
	list_for_each_entry(slot, &text_ksm_mm_head.slot.mm_node, mm_node) {
		mm_slot = mm_slot_entry(slot, struct text_ksm_mm_slot, slot);
		mm_slot->scan_done = false;
	}
	spin_unlock(&text_ksm_mmlist_lock);

	text_ksm_total_acc = 0;
	text_ksm_resident_acc = 0;
	text_ksm_mergeable_acc = 0;

	text_ksm_scan.mm = NULL;
	text_ksm_scan.inode = NULL;
	text_ksm_scan.address = 0;

	text_ksm_scan_active = true;
}

/*
 * text_ksm_scan_finish - end the current scan: free the seen-inode list, drop
 * the cursor's mm reference, and clear the active flag. If @publish is true
 * (the scan completed naturally), the accumulated counters are published to
 * the sysfs attributes; if false (the scan was torn down mid-way, e.g. run
 * turned off), the last completed scan's values are kept.
 */
static void text_ksm_scan_finish(bool publish)
{
	struct text_ksm_seen_inode *si, *tmp;

	text_ksm_stable_tree_free();
	root_text_ksm_stable_tree = RB_ROOT;

	list_for_each_entry_safe(si, tmp, &text_ksm_seen_inodes, list) {
		list_del(&si->list);
		iput(si->inode);
		kfree(si);
	}

	if (text_ksm_scan.mm) {
		mmdrop(text_ksm_scan.mm);
		text_ksm_scan.mm = NULL;
	}
	text_ksm_scan.inode = NULL;
	text_ksm_scan.address = 0;

	if (publish) {
		/*
		 * Publish the completed scan's result to the sysfs counters.
		 * Done here (not live) so the counters always reflect a full
		 * scan and stay stable across the scan's per-cycle restarts.
		 */
		text_ksm_pages_total = text_ksm_total_acc;
		text_ksm_pages_resident = text_ksm_resident_acc;
		text_ksm_pages_mergeable = text_ksm_mergeable_acc;
	}

	text_ksm_scan_active = false;
}

/*
 * text_ksm_do_scan - detection scan for file-backed read-only executable
 * pages.
 *
 * Stateful and budget-limited: scans at most @scan_npages page-cache indices
 * and resumes where it left off on the next call, so ktextksmd's CPU usage
 * stays bounded by text_ksm_pages_to_scan.
 *
 * The scan walks the text_ksm_mm_head list (the mms that opted in via
 * prctl(PR_SET_TEXT_MERGE)) rather than for_each_process. For each eligible
 * file VMA (PROT_EXEC, no VM_WRITE, inode passing the read-only predicate) it
 * walks the page cache and counts the resident single-page folios. Each
 * inode's page cache is walked once per scan (deduped), so a file mapped by
 * many mms is not counted more than once.
 */
static void text_ksm_do_scan(unsigned int scan_npages)
{
	if (!text_ksm_scan_active)
		text_ksm_scan_start();

	while (scan_npages-- && likely(!freezing(current))) {
		cond_resched();
		if (!text_ksm_scan_next()) {
			text_ksm_scan_finish(true);
			return;
		}
	}
}

static bool text_ksm_should_run(void)
{
	return text_ksm_run && !list_empty(&text_ksm_mm_head.slot.mm_node);
}

static int text_ksm_scan_thread(void *nothing)
{
	unsigned int sleep_ms;

	set_freezable();
	set_user_nice(current, 5);

	while (!kthread_should_stop()) {
		bool run = READ_ONCE(text_ksm_run);

		mutex_lock(&text_ksm_thread_mutex);
		if (text_ksm_should_run())
			text_ksm_do_scan(text_ksm_pages_to_scan);
		else if (text_ksm_scan_active)
			/*
			 * run was turned off mid-scan: tear down without
			 * publishing (the scan is incomplete).
			 */
			text_ksm_scan_finish(false);
		mutex_unlock(&text_ksm_thread_mutex);

		if (text_ksm_should_run() || run) {
			sleep_ms = READ_ONCE(text_ksm_sleep_millisecs);
			wait_event_freezable_timeout(text_ksm_iter_wait,
				sleep_ms != READ_ONCE(text_ksm_sleep_millisecs) ||
				!READ_ONCE(text_ksm_run),
				msecs_to_jiffies(sleep_ms));
		} else {
			wait_event_freezable(text_ksm_thread_wait,
				text_ksm_should_run() ||
				READ_ONCE(text_ksm_run) ||
				kthread_should_stop());
		}
	}
	return 0;
}

/*
 * __text_ksm_enter - add @mm to the text_ksm mm list.
 *
 * Called by the prctl opt-in and by text_ksm_fork(). Takes a reference on the
 * mm (mmgrab) so it stays alive while on the list; __text_ksm_exit() drops it.
 *
 * Returns 0 on success, -ENOMEM on allocation failure.
 */
int __text_ksm_enter(struct mm_struct *mm)
{
	struct text_ksm_mm_slot *mm_slot;
	struct mm_slot *slot;
	int needs_wakeup;

	mm_slot = mm_slot_alloc(text_ksm_mm_slot_cache);
	if (!mm_slot)
		return -ENOMEM;

	slot = &mm_slot->slot;

	spin_lock(&text_ksm_mmlist_lock);
	needs_wakeup = list_empty(&text_ksm_mm_head.slot.mm_node);
	mm_slot_insert(text_ksm_mm_slots_hash, mm, slot);
	list_add_tail(&slot->mm_node, &text_ksm_mm_head.slot.mm_node);
	spin_unlock(&text_ksm_mmlist_lock);

	mm_flags_set(MMF_TEXT_MERGEABLE, mm);
	mmgrab(mm);

	if (needs_wakeup)
		wake_up_interruptible(&text_ksm_thread_wait);

	return 0;
}

/*
 * __text_ksm_exit - remove @mm from the text_ksm mm list.
 *
 * Idempotent: if @mm is not on the list (e.g. already removed by a racing
 * prctl disable or a previous exit), it does nothing. The slot removal and
 * the mm reference drop are paired, so a double call cannot double-free or
 * unbalance the mm reference.
 */
void __text_ksm_exit(struct mm_struct *mm)
{
	struct text_ksm_mm_slot *mm_slot;
	struct mm_slot *slot;

	spin_lock(&text_ksm_mmlist_lock);
	slot = mm_slot_lookup(text_ksm_mm_slots_hash, mm);
	if (slot) {
		mm_slot = mm_slot_entry(slot, struct text_ksm_mm_slot, slot);
		mm_slot_remove(slot);
	}
	spin_unlock(&text_ksm_mmlist_lock);

	if (slot) {
		mm_flags_clear(MMF_TEXT_MERGEABLE, mm);
		mm_slot_free(text_ksm_mm_slot_cache, mm_slot);
		mmdrop(mm);
	}
}

/*
 * text_ksm_enable - opt @mm in for text_ksm merging (prctl(PR_SET_TEXT_MERGE,
 * 1)).
 *
 * Returns 0 on success, otherwise an error code.
 */
int text_ksm_enable(struct mm_struct *mm)
{
	if (mm_flags_test(MMF_TEXT_MERGEABLE, mm))
		return 0;

	return __text_ksm_enter(mm);
}

/*
 * text_ksm_disable - opt @mm out of text_ksm merging (prctl(PR_SET_TEXT_MERGE,
 * 0)).
 *
 * Returns 0 on success.
 */
int text_ksm_disable(struct mm_struct *mm)
{
	if (!mm_flags_test(MMF_TEXT_MERGEABLE, mm))
		return 0;

	__text_ksm_exit(mm);
	return 0;
}

#ifdef CONFIG_SYSFS
/*
 * This all compiles without CONFIG_SYSFS, but is a waste of space.
 */

#define TEXT_KSM_ATTR_RO(_name) \
	static struct kobj_attribute _name##_attr = __ATTR_RO(_name)
#define TEXT_KSM_ATTR(_name) \
	static struct kobj_attribute _name##_attr = __ATTR_RW(_name)

static ssize_t run_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	return sysfs_emit(buf, "%u\n", text_ksm_run);
}

static ssize_t run_store(struct kobject *kobj, struct kobj_attribute *attr,
			 const char *buf, size_t count)
{
	int err;
	bool value;

	err = kstrtobool(buf, &value);
	if (err)
		return -EINVAL;

	mutex_lock(&text_ksm_thread_mutex);
	text_ksm_run = value;
	mutex_unlock(&text_ksm_thread_mutex);

	if (value)
		wake_up_interruptible(&text_ksm_thread_wait);
	wake_up_interruptible(&text_ksm_iter_wait);

	return count;
}
TEXT_KSM_ATTR(run);

static ssize_t pages_to_scan_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", text_ksm_pages_to_scan);
}

static ssize_t pages_to_scan_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int nr_pages;
	int err;

	err = kstrtouint(buf, 10, &nr_pages);
	if (err)
		return -EINVAL;

	text_ksm_pages_to_scan = nr_pages;

	return count;
}
TEXT_KSM_ATTR(pages_to_scan);

static ssize_t sleep_millisecs_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", text_ksm_sleep_millisecs);
}

static ssize_t sleep_millisecs_store(struct kobject *kobj,
				     struct kobj_attribute *attr,
				     const char *buf, size_t count)
{
	unsigned int msecs;
	int err;

	err = kstrtouint(buf, 10, &msecs);
	if (err)
		return -EINVAL;

	text_ksm_sleep_millisecs = msecs;
	wake_up_interruptible(&text_ksm_iter_wait);

	return count;
}
TEXT_KSM_ATTR(sleep_millisecs);

static ssize_t pages_total_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", text_ksm_pages_total);
}
TEXT_KSM_ATTR_RO(pages_total);

static ssize_t pages_resident_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", text_ksm_pages_resident);
}
TEXT_KSM_ATTR_RO(pages_resident);

static ssize_t pages_mergeable_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", text_ksm_pages_mergeable);
}
TEXT_KSM_ATTR_RO(pages_mergeable);

static struct attribute *text_ksm_attrs[] = {
	&run_attr.attr,
	&pages_to_scan_attr.attr,
	&sleep_millisecs_attr.attr,
	&pages_total_attr.attr,
	&pages_resident_attr.attr,
	&pages_mergeable_attr.attr,
	NULL,
};

static const struct attribute_group text_ksm_attr_group = {
	.attrs = text_ksm_attrs,
};

static struct kobject *text_ksm_kobj;
#endif /* CONFIG_SYSFS */

static int __init text_ksm_init(void)
{
	struct task_struct *text_ksm_thread;
	int err;

	text_ksm_mm_slot_cache = KMEM_CACHE(text_ksm_mm_slot, 0);
	if (!text_ksm_mm_slot_cache)
		return -ENOMEM;

	text_ksm_thread = kthread_run(text_ksm_scan_thread, NULL, "ktextksmd");
	if (IS_ERR(text_ksm_thread)) {
		pr_err("text_ksm: creating kthread failed\n");
		err = PTR_ERR(text_ksm_thread);
		goto out_free;
	}

#ifdef CONFIG_SYSFS
	text_ksm_kobj = kobject_create_and_add("text_ksm", mm_kobj);
	if (!text_ksm_kobj) {
		pr_err("text_ksm: creating kobject failed\n");
		err = -ENOMEM;
		goto out_thread;
	}
	err = sysfs_create_group(text_ksm_kobj, &text_ksm_attr_group);
	if (err) {
		pr_err("text_ksm: register sysfs failed\n");
		goto out_kobj;
	}
#else
	text_ksm_run = true;	/* no way for user to start it */
#endif /* CONFIG_SYSFS */

	return 0;

#ifdef CONFIG_SYSFS
out_kobj:
	kobject_put(text_ksm_kobj);
out_thread:
	kthread_stop(text_ksm_thread);
#endif
out_free:
	kmem_cache_destroy(text_ksm_mm_slot_cache);
	text_ksm_mm_slot_cache = NULL;
	return err;
}
subsys_initcall(text_ksm_init);
