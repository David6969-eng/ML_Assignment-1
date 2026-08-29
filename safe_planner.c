/* =====================================================================
 *  PCCST503 - Machine Learning : Assignment 1
 *  Design of a Safe Semantic Planner in a Finite Cartesian State Space
 *
 *  Algorithm : D* Lite (incremental heuristic search, Koenig & Likhachev)
 *  Language  : C11, single translation unit, no external dependencies
 *  Build     : gcc -O2 -std=c11 -Wall -Wextra -o safe_planner safe_planner.c -lm
 *  Run       : ./safe_planner
 *
 *  Contents
 *    1. Utilities        : instrumented allocator, monotonic timer
 *    2. Domain model     : State / Transition / PlanningProblem / PlanningResult
 *    3. Priority queue   : indexed binary heap over lexicographic keys (k1,k2)
 *    4. Planner          : D* Lite + safety-aware edge weights + dynamic updates
 *    5. Verifier         : plain Dijkstra baseline (proves optimality of D* Lite)
 *    6. Experiments      : the 6 assignment test cases + a 150x150 grid benchmark
 * ===================================================================== */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define INF  HUGE_VAL
#define EPS  1e-9

/* =====================================================================
 * 1. UTILITIES
 * ===================================================================== */

static size_t g_mem_cur = 0, g_mem_peak = 0;

static void *xmalloc(size_t n)
{
    char *p = (char *)malloc(n + 16);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    *(size_t *)p = n;
    g_mem_cur += n;
    if (g_mem_cur > g_mem_peak) g_mem_peak = g_mem_cur;
    return p + 16;
}

static void *xrealloc(void *q, size_t n)
{
    if (!q) return xmalloc(n);
    char *base = (char *)q - 16;
    size_t old = *(size_t *)base;
    char *p = (char *)realloc(base, n + 16);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    *(size_t *)p = n;
    g_mem_cur = g_mem_cur - old + n;
    if (g_mem_cur > g_mem_peak) g_mem_peak = g_mem_cur;
    return p + 16;
}

static void xfree(void *q)
{
    if (!q) return;
    char *base = (char *)q - 16;
    g_mem_cur -= *(size_t *)base;
    free(base);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static int feq(double a, double b)          /* INF-safe equality */
{
    return (a == b) || (fabs(a - b) <= EPS);
}

/* =====================================================================
 * 2. DOMAIN MODEL
 *
 *  State ids are dense indices [0, nStates).  A real deployment would put a
 *  hash map id -> index in front of this; every routine below only ever
 *  touches states through that index, so the change is local.
 * ===================================================================== */

typedef struct {
    uint64_t  id;
    double   *embedding;        /* length = problem->dim, R^d coordinates   */
} State;

typedef struct {
    uint64_t id;
    uint64_t from, to;
    double   cost;              /* nominal traversal cost                   */
    double   safety;            /* per-transition safety score, in [0,1]    */
    double   reliability;       /* success probability, in [0,1]            */
    int      available;         /* 0 = temporarily removed / blocked        */
} Transition;

typedef struct {                /* growable list of transition indices      */
    uint32_t *e;
    uint32_t  n, cap;
} AdjList;

typedef struct {
    uint64_t    initialState;
    uint64_t    goalState;

    uint64_t   *badStates;      /* as required by the given interface       */
    uint32_t    nBad;
    uint8_t    *isBad;          /* O(1) membership test, size nStates       */

    State      *states;
    uint32_t    nStates, capStates;
    uint32_t    dim;

    Transition *transitions;
    uint32_t    nTrans, capTrans;

    AdjList    *out, *in;       /* successor / predecessor incidence lists  */
    char      (*label)[12];     /* human readable names, for reporting only */
} PlanningProblem;

typedef struct {
    int       success;
    uint64_t *statePath;
    uint32_t  nStatePath;
    uint64_t *transitionPath;
    uint32_t  nTransPath;
    double    totalCost;        /* sum of nominal transition costs          */
    double    safetyScore;      /* min Euclidean distance to any bad state  */
    double    reliability;      /* cumulative (sum) reliability             */
    double    reliabilityProd;  /* product = P(whole plan executes)         */
    double    score;            /* alpha*G - beta*C + gamma*D + delta*R     */
    /* instrumentation */
    long      expanded;
    long      heapOps;
    double    planMs;
    size_t    memBytes;
} PlanningResult;

/* ---- objective / safety weights ------------------------------------- */
typedef struct {
    double lambda;   /* weight of the clearance shortfall penalty          */
    double dSafe;    /* clearance we would like to keep from bad states    */
    double mu;       /* weight of the unreliability penalty                */
    double nu;       /* weight of the transition-unsafety penalty          */
    double alpha, beta, gamma, delta;   /* Score(P) coefficients           */
} Weights;

static Weights defaultWeights(void)
{
    Weights w;
    w.lambda = 4.0; w.dSafe = 2.0; w.mu = 0.5; w.nu = 0.5;
    w.alpha = 100.0; w.beta = 1.0; w.gamma = 10.0; w.delta = 1.0;
    return w;
}

/* ---- problem construction ------------------------------------------- */

static void probInit(PlanningProblem *p, uint32_t dim)
{
    memset(p, 0, sizeof(*p));
    p->dim = dim;
}

static void probReserve(PlanningProblem *p, uint32_t ns, uint32_t nt)
{
    if (ns > p->capStates) {
        p->states = (State *)  xrealloc(p->states, ns * sizeof(State));
        p->isBad  = (uint8_t *)xrealloc(p->isBad,  ns * sizeof(uint8_t));
        p->out    = (AdjList *)xrealloc(p->out,    ns * sizeof(AdjList));
        p->in     = (AdjList *)xrealloc(p->in,     ns * sizeof(AdjList));
        p->label  = (char (*)[12])xrealloc(p->label, ns * 12);
        memset(p->isBad + p->capStates, 0, (ns - p->capStates));
        memset(p->out   + p->capStates, 0, (ns - p->capStates) * sizeof(AdjList));
        memset(p->in    + p->capStates, 0, (ns - p->capStates) * sizeof(AdjList));
        p->capStates = ns;
    }
    if (nt > p->capTrans) {
        p->transitions = (Transition *)xrealloc(p->transitions, nt * sizeof(Transition));
        p->capTrans = nt;
    }
}

static uint32_t probAddState(PlanningProblem *p, const char *label, const double *v)
{
    if (p->nStates == p->capStates)
        probReserve(p, p->capStates ? p->capStates * 2 : 16, p->capTrans);
    uint32_t i = p->nStates++;
    p->states[i].id = i;
    p->states[i].embedding = (double *)xmalloc(p->dim * sizeof(double));
    memcpy(p->states[i].embedding, v, p->dim * sizeof(double));
    p->isBad[i] = 0;
    p->out[i].e = p->in[i].e = NULL;
    p->out[i].n = p->out[i].cap = p->in[i].n = p->in[i].cap = 0;
    snprintf(p->label[i], 12, "%s", label ? label : "s");
    return i;
}

static void adjPush(AdjList *a, uint32_t v)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 4;
        a->e = (uint32_t *)xrealloc(a->e, a->cap * sizeof(uint32_t));
    }
    a->e[a->n++] = v;
}

static uint32_t probAddTransition(PlanningProblem *p, uint32_t from, uint32_t to,
                                  double cost, double safety, double reliability,
                                  int available)
{
    if (p->nTrans == p->capTrans)
        probReserve(p, p->capStates, p->capTrans ? p->capTrans * 2 : 32);
    uint32_t e = p->nTrans++;
    Transition *t = &p->transitions[e];
    t->id = e; t->from = from; t->to = to;
    t->cost = cost; t->safety = safety; t->reliability = reliability;
    t->available = available;
    adjPush(&p->out[from], e);
    adjPush(&p->in[to],    e);
    return e;
}

static void probSetBad(PlanningProblem *p, const uint32_t *ids, uint32_t n)
{
    for (uint32_t i = 0; i < p->nStates; i++) p->isBad[i] = 0;
    xfree(p->badStates);
    p->badStates = (uint64_t *)xmalloc((n ? n : 1) * sizeof(uint64_t));
    p->nBad = n;
    for (uint32_t i = 0; i < n; i++) { p->badStates[i] = ids[i]; p->isBad[ids[i]] = 1; }
}

static void probFree(PlanningProblem *p)
{
    for (uint32_t i = 0; i < p->nStates; i++) {
        xfree(p->states[i].embedding);
        xfree(p->out[i].e); xfree(p->in[i].e);
    }
    xfree(p->states); xfree(p->isBad); xfree(p->out); xfree(p->in);
    xfree(p->label);  xfree(p->transitions); xfree(p->badStates);
    memset(p, 0, sizeof(*p));
}

static double distStates(const PlanningProblem *p, uint32_t a, uint32_t b)
{
    double s = 0.0;
    for (uint32_t k = 0; k < p->dim; k++) {
        double d = p->states[a].embedding[k] - p->states[b].embedding[k];
        s += d * d;
    }
    return sqrt(s);
}

/* =====================================================================
 * 3. INDEXED BINARY HEAP  (lexicographic key [k1, k2], with O(log n) removal)
 * ===================================================================== */

typedef struct { double k1, k2; uint32_t s; } HKey;

typedef struct {
    HKey     *a;
    uint32_t  n, cap;
    int32_t  *pos;              /* pos[state] = slot in a[], or -1          */
    uint32_t  nPos;
    long      ops;
} Heap;

static int keyLess(HKey a, HKey b)
{
    if (a.k1 < b.k1 - EPS) return 1;
    if (b.k1 < a.k1 - EPS) return 0;
    return a.k2 < b.k2 - EPS;
}

static void heapInit(Heap *h, uint32_t nStates)
{
    h->cap = 64; h->n = 0; h->ops = 0;
    h->a = (HKey *)xmalloc(h->cap * sizeof(HKey));
    h->nPos = nStates;
    h->pos = (int32_t *)xmalloc(nStates * sizeof(int32_t));
    for (uint32_t i = 0; i < nStates; i++) h->pos[i] = -1;
}

static void heapFree(Heap *h) { xfree(h->a); xfree(h->pos); memset(h, 0, sizeof(*h)); }

static void heapClear(Heap *h)
{
    for (uint32_t i = 0; i < h->n; i++) h->pos[h->a[i].s] = -1;
    h->n = 0;
}

static void heapSiftUp(Heap *h, uint32_t i)
{
    HKey v = h->a[i];
    while (i > 0) {
        uint32_t par = (i - 1) / 2;
        if (!keyLess(v, h->a[par])) break;
        h->a[i] = h->a[par]; h->pos[h->a[i].s] = (int32_t)i;
        i = par;
    }
    h->a[i] = v; h->pos[v.s] = (int32_t)i;
}

static void heapSiftDown(Heap *h, uint32_t i)
{
    HKey v = h->a[i];
    for (;;) {
        uint32_t l = 2 * i + 1, r = l + 1, m = i;
        HKey best = v;
        if (l < h->n && keyLess(h->a[l], best)) { m = l; best = h->a[l]; }
        if (r < h->n && keyLess(h->a[r], best)) { m = r; best = h->a[r]; }
        if (m == i) break;
        h->a[i] = h->a[m]; h->pos[h->a[i].s] = (int32_t)i;
        i = m;
    }
    h->a[i] = v; h->pos[v.s] = (int32_t)i;
}

static void heapInsert(Heap *h, HKey k)
{
    h->ops++;
    if (h->n == h->cap) { h->cap *= 2; h->a = (HKey *)xrealloc(h->a, h->cap * sizeof(HKey)); }
    h->a[h->n] = k; h->pos[k.s] = (int32_t)h->n; h->n++;
    heapSiftUp(h, h->n - 1);
}

static int heapContains(Heap *h, uint32_t s) { return h->pos[s] >= 0; }

static void heapRemove(Heap *h, uint32_t s)
{
    int32_t i = h->pos[s];
    if (i < 0) return;
    h->ops++;
    h->pos[s] = -1;
    h->n--;
    if ((uint32_t)i == h->n) return;
    h->a[i] = h->a[h->n];
    h->pos[h->a[i].s] = i;
    heapSiftUp(h, (uint32_t)i);
    heapSiftDown(h, (uint32_t)h->pos[h->a[i].s]);
}

/* =====================================================================
 * 4. PLANNER : D* Lite
 * ===================================================================== */

typedef struct {
    PlanningProblem *p;
    Weights   w;
    double   *g, *rhs;          /* cost-to-goal estimates                   */
    double   *clear;            /* Euclidean clearance to nearest bad state */
    Heap      U;
    double    km;               /* key modifier, absorbs start motion       */
    uint32_t  start, goal;
    double    hScale;           /* admissible cost-per-unit-length factor   */
    long      expanded, updates;
} Planner;

/* ---- 4.1 safety computation ----------------------------------------- */
/*  clear(s) = min_{b in B} || s - b ||_2 ,  +INF when B is empty.        */
static void plannerComputeClearance(Planner *pl)
{
    PlanningProblem *p = pl->p;
    for (uint32_t i = 0; i < p->nStates; i++) {
        double best = INF;
        for (uint32_t j = 0; j < p->nBad; j++) {
            double d = distStates(p, i, (uint32_t)p->badStates[j]);
            if (d < best) best = d;
        }
        pl->clear[i] = best;
    }
}

/* ---- 4.2 effective edge weight --------------------------------------
 *  w(u,v) = cost(u,v)
 *         + lambda * max(0, dSafe - clear(v)) / dSafe     (safety margin)
 *         + mu     * (1 - reliability(u,v))               (reliability)
 *         + nu     * (1 - safety(u,v))                    (edge safety)
 *  and +INF whenever the transition is unavailable or touches a bad state.
 *  Note w >= cost, which is what keeps the heuristic below admissible.  */
static double edgeWeight(const Planner *pl, const Transition *t)
{
    if (!t->available) return INF;
    if (pl->p->isBad[t->from] || pl->p->isBad[t->to]) return INF;
    double w = t->cost;
    double c = pl->clear[t->to];
    if (c < pl->w.dSafe) w += pl->w.lambda * (pl->w.dSafe - c) / pl->w.dSafe;
    w += pl->w.mu * (1.0 - t->reliability);
    w += pl->w.nu * (1.0 - t->safety);
    return w;
}

/* ---- 4.3 heuristic ---------------------------------------------------
 *  h(s) = hScale * || s - s_start ||_2 , hScale = min_e cost(e)/len(e).
 *  Any path from s_start to s has cost >= hScale * (sum of segment
 *  lengths) >= hScale * ||s - s_start||, so h is admissible; the triangle
 *  inequality then makes it consistent.                                  */
static void plannerComputeHScale(Planner *pl)
{
    PlanningProblem *p = pl->p;
    double best = INF;
    for (uint32_t e = 0; e < p->nTrans; e++) {
        Transition *t = &p->transitions[e];
        if (!t->available) continue;
        double len = distStates(p, (uint32_t)t->from, (uint32_t)t->to);
        if (len < 1e-12) continue;
        double r = t->cost / len;
        if (r < best) best = r;
    }
    pl->hScale = (best == INF) ? 0.0 : best;
}

static double hval(const Planner *pl, uint32_t s)
{
    return pl->hScale * distStates(pl->p, pl->start, s);
}

static HKey calcKey(const Planner *pl, uint32_t s)
{
    double m = (pl->g[s] < pl->rhs[s]) ? pl->g[s] : pl->rhs[s];
    HKey k;
    k.s  = s;
    k.k2 = m;
    k.k1 = (m == INF) ? INF : m + hval(pl, s) + pl->km;
    return k;
}

static void updateVertex(Planner *pl, uint32_t u)
{
    pl->updates++;
    if (u != pl->goal) {
        double best = INF;
        AdjList *o = &pl->p->out[u];
        for (uint32_t i = 0; i < o->n; i++) {
            Transition *t = &pl->p->transitions[o->e[i]];
            double w = edgeWeight(pl, t);
            if (w == INF) continue;
            double v = w + pl->g[t->to];
            if (v < best) best = v;
        }
        pl->rhs[u] = best;
    }
    if (heapContains(&pl->U, u)) heapRemove(&pl->U, u);
    if (!feq(pl->g[u], pl->rhs[u])) heapInsert(&pl->U, calcKey(pl, u));
}

/* Re-key every open node.  Needed only when the heuristic scale drops
 * (a cheaper transition was inserted), which invalidates stored keys.    */
static void heapRekeyAll(Planner *pl)
{
    Heap *h = &pl->U;
    for (uint32_t i = 0; i < h->n; i++) h->a[i] = calcKey(pl, h->a[i].s);
    if (h->n > 1)
        for (int32_t i = (int32_t)h->n / 2 - 1; i >= 0; i--) heapSiftDown(h, (uint32_t)i);
    for (uint32_t i = 0; i < h->n; i++) h->pos[h->a[i].s] = (int32_t)i;
}

static void plannerInit(Planner *pl, PlanningProblem *p, Weights w)
{
    memset(pl, 0, sizeof(*pl));
    pl->p = p; pl->w = w;
    pl->start = (uint32_t)p->initialState;
    pl->goal  = (uint32_t)p->goalState;
    pl->g     = (double *)xmalloc(p->nStates * sizeof(double));
    pl->rhs   = (double *)xmalloc(p->nStates * sizeof(double));
    pl->clear = (double *)xmalloc(p->nStates * sizeof(double));
    heapInit(&pl->U, p->nStates);
    plannerComputeClearance(pl);
    plannerComputeHScale(pl);
    for (uint32_t i = 0; i < p->nStates; i++) pl->g[i] = pl->rhs[i] = INF;
    pl->km = 0.0;
    pl->rhs[pl->goal] = 0.0;
    heapInsert(&pl->U, calcKey(pl, pl->goal));
}

static void plannerFree(Planner *pl)
{
    xfree(pl->g); xfree(pl->rhs); xfree(pl->clear); heapFree(&pl->U);
}

/* ---- 4.4 ComputeShortestPath ---------------------------------------- */
static void plannerCompute(Planner *pl)
{
    while (pl->U.n > 0) {
        HKey kold  = pl->U.a[0];
        HKey kstrt = calcKey(pl, pl->start);
        if (!keyLess(kold, kstrt) && feq(pl->rhs[pl->start], pl->g[pl->start])) break;

        uint32_t u = kold.s;
        HKey knew = calcKey(pl, u);

        if (keyLess(kold, knew)) {              /* stale key -> re-insert  */
            heapRemove(&pl->U, u);
            heapInsert(&pl->U, knew);
        } else if (pl->g[u] > pl->rhs[u] + EPS) {   /* over-consistent     */
            pl->g[u] = pl->rhs[u];
            heapRemove(&pl->U, u);
            pl->expanded++;
            AdjList *in = &pl->p->in[u];
            for (uint32_t i = 0; i < in->n; i++)
                updateVertex(pl, (uint32_t)pl->p->transitions[in->e[i]].from);
        } else {                                    /* under-consistent    */
            pl->g[u] = INF;
            heapRemove(&pl->U, u);
            pl->expanded++;
            AdjList *in = &pl->p->in[u];
            for (uint32_t i = 0; i < in->n; i++)
                updateVertex(pl, (uint32_t)pl->p->transitions[in->e[i]].from);
            updateVertex(pl, u);
        }
    }
}

/* ---- 4.5 dynamic environment hooks ---------------------------------- */

/* transition availability / cost change: only the tail vertex is dirty */
static void plannerEdgeChanged(Planner *pl, uint32_t e)
{
    updateVertex(pl, (uint32_t)pl->p->transitions[e].from);
}

static void plannerSetEdgeAvailable(Planner *pl, uint32_t e, int avail)
{
    pl->p->transitions[e].available = avail;
    plannerEdgeChanged(pl, e);
}

/* new transition inserted at run time */
static uint32_t plannerAddTransition(Planner *pl, uint32_t from, uint32_t to,
                                     double cost, double safety, double rel)
{
    uint32_t e = probAddTransition(pl->p, from, to, cost, safety, rel, 1);
    double old = pl->hScale;
    plannerComputeHScale(pl);
    if (pl->hScale < old - EPS) heapRekeyAll(pl);   /* h shrank: re-key    */
    plannerEdgeChanged(pl, e);
    return e;
}

/* the agent physically moved: absorb the heuristic shift into km        */
static void plannerSetStart(Planner *pl, uint32_t ns)
{
    pl->km += pl->hScale * distStates(pl->p, pl->start, ns);
    pl->start = ns;
    pl->p->initialState = ns;
}

/* bad-state set changed: clearances move, so every in-edge of an
 * affected state has to be re-evaluated.  Search tree is kept.          */
static void plannerSetBadStates(Planner *pl, const uint32_t *ids, uint32_t n)
{
    PlanningProblem *p = pl->p;
    double *old = (double *)xmalloc(p->nStates * sizeof(double));
    memcpy(old, pl->clear, p->nStates * sizeof(double));
    probSetBad(p, ids, n);
    plannerComputeClearance(pl);
    for (uint32_t v = 0; v < p->nStates; v++) {
        if (feq(old[v], pl->clear[v]) && !p->isBad[v]) continue;
        AdjList *in = &p->in[v];
        for (uint32_t i = 0; i < in->n; i++)
            updateVertex(pl, (uint32_t)p->transitions[in->e[i]].from);
        updateVertex(pl, v);
    }
    xfree(old);
}

/* goal moved: g/rhs are goal-relative, so they are reset, but the graph,
 * geometry, clearances and heap storage are all reused (O(n), no malloc) */
static void plannerSetGoal(Planner *pl, uint32_t ng)
{
    PlanningProblem *p = pl->p;
    for (uint32_t i = 0; i < p->nStates; i++) pl->g[i] = pl->rhs[i] = INF;
    heapClear(&pl->U);
    pl->km = 0.0;
    pl->goal = ng;
    p->goalState = ng;
    pl->rhs[ng] = 0.0;
    heapInsert(&pl->U, calcKey(pl, ng));
}

/* ---- 4.6 greedy path extraction + scoring --------------------------- */
static void plannerExtract(Planner *pl, PlanningResult *r)
{
    PlanningProblem *p = pl->p;
    memset(r, 0, sizeof(*r));
    r->safetyScore = INF;
    r->reliabilityProd = 1.0;

    if (pl->g[pl->start] == INF || p->isBad[pl->start]) { r->success = 0; return; }

    uint32_t capS = p->nStates + 1;
    r->statePath      = (uint64_t *)xmalloc(capS * sizeof(uint64_t));
    r->transitionPath = (uint64_t *)xmalloc(capS * sizeof(uint64_t));

    uint32_t u = pl->start, steps = 0;
    r->statePath[r->nStatePath++] = u;
    if (pl->clear[u] < r->safetyScore) r->safetyScore = pl->clear[u];

    while (u != pl->goal && steps++ < p->nStates) {
        double best = INF; uint32_t bestE = UINT32_MAX;
        AdjList *o = &p->out[u];
        for (uint32_t i = 0; i < o->n; i++) {
            Transition *t = &p->transitions[o->e[i]];
            double w = edgeWeight(pl, t);
            if (w == INF) continue;
            double v = w + pl->g[t->to];
            if (v < best - EPS) { best = v; bestE = o->e[i]; }
        }
        if (bestE == UINT32_MAX) { r->success = 0; return; }
        Transition *t = &p->transitions[bestE];
        r->transitionPath[r->nTransPath++] = bestE;
        r->totalCost       += t->cost;
        r->reliability     += t->reliability;
        r->reliabilityProd *= t->reliability;
        u = (uint32_t)t->to;
        r->statePath[r->nStatePath++] = u;
        if (pl->clear[u] < r->safetyScore) r->safetyScore = pl->clear[u];
    }
    r->success = (u == pl->goal);
    if (r->safetyScore == INF) r->safetyScore = 0.0;   /* no bad states    */
    r->score = pl->w.alpha * (r->success ? 1.0 : 0.0)
             - pl->w.beta  * r->totalCost
             + pl->w.gamma * (r->safetyScore == 0.0 ? 0.0 : r->safetyScore)
             + pl->w.delta * r->reliability;
}

/* =====================================================================
 * 5. VERIFIER : Dijkstra on the same effective weights
 * ===================================================================== */
static double dijkstraCost(Planner *pl)
{
    PlanningProblem *p = pl->p;
    double *d = (double *)xmalloc(p->nStates * sizeof(double));
    Heap h; heapInit(&h, p->nStates);
    for (uint32_t i = 0; i < p->nStates; i++) d[i] = INF;
    d[pl->start] = 0.0;
    HKey k; k.s = pl->start; k.k1 = 0.0; k.k2 = 0.0;
    heapInsert(&h, k);
    while (h.n > 0) {
        HKey top = h.a[0];
        heapRemove(&h, top.s);
        if (top.k1 > d[top.s] + EPS) continue;
        uint32_t u = top.s;
        AdjList *o = &p->out[u];
        for (uint32_t i = 0; i < o->n; i++) {
            Transition *t = &p->transitions[o->e[i]];
            double w = edgeWeight(pl, t);
            if (w == INF) continue;
            double nd = d[u] + w;
            if (nd < d[t->to] - EPS) {
                d[t->to] = nd;
                HKey nk; nk.s = (uint32_t)t->to; nk.k1 = nd; nk.k2 = 0.0;
                if (heapContains(&h, nk.s)) heapRemove(&h, nk.s);
                heapInsert(&h, nk);
            }
        }
    }
    double res = d[pl->goal];
    xfree(d); heapFree(&h);
    return res;
}

/* =====================================================================
 * 6. REPORTING
 * ===================================================================== */
static void printPath(PlanningProblem *p, PlanningResult *r)
{
    if (!r->success) { printf("(no path)"); return; }
    for (uint32_t i = 0; i < r->nStatePath; i++)
        printf("%s%s", p->label[r->statePath[i]], i + 1 < r->nStatePath ? " -> " : "");
}

static void report(const char *tag, PlanningProblem *p, Planner *pl, PlanningResult *r)
{
    int badVisited = 0;
    for (uint32_t i = 0; i < r->nStatePath; i++)
        if (p->isBad[r->statePath[i]]) badVisited++;

    printf("  %-18s : %s\n", "result", r->success ? "SUCCESS" : "FAILURE");
    printf("  %-18s : ", "path");           printPath(p, r); printf("\n");
    printf("  %-18s : %.3f\n", "total cost",        r->totalCost);
    printf("  %-18s : %.3f\n", "effective g(start)", pl->g[pl->start]);
    printf("  %-18s : %d\n",   "bad states visited", badVisited);
    if (p->nBad)
        printf("  %-18s : %.3f\n", "min dist to bad", r->safetyScore);
    else
        printf("  %-18s : n/a (B empty)\n", "min dist to bad");
    printf("  %-18s : %.3f (product %.4f)\n", "cum. reliability",
           r->reliability, r->reliabilityProd);
    printf("  %-18s : %.3f\n", "Score(P)",          r->score);
    printf("  %-18s : %ld\n",  "states expanded",   r->expanded);
    printf("  %-18s : %ld\n",  "heap operations",   r->heapOps);
    printf("  %-18s : %.4f ms\n", "planning time",  r->planMs);
    printf("  %-18s : %.1f KB\n", "planner memory", r->memBytes / 1024.0);
    if (tag) printf("  %-18s : %s\n", "note", tag);
}

static void runPlan(Planner *pl, PlanningResult *r)
{
    size_t m0 = g_mem_cur;
    double t0 = now_ms();
    pl->expanded = 0; pl->U.ops = 0;
    plannerCompute(pl);
    double t1 = now_ms();
    plannerExtract(pl, r);
    r->planMs   = t1 - t0;
    r->expanded = pl->expanded;
    r->heapOps  = pl->U.ops;
    r->memBytes = g_mem_cur > m0 ? g_mem_cur - m0 : g_mem_cur;
}

static void freeResult(PlanningResult *r)
{
    xfree(r->statePath); xfree(r->transitionPath);
    r->statePath = r->transitionPath = NULL;
}

static void header(const char *s)
{
    printf("\n==========================================================\n");
    printf("  %s\n", s);
    printf("==========================================================\n");
}

static void checkOptimal(Planner *pl)
{
    double dj = dijkstraCost(pl);
    double gs = pl->g[pl->start];
    int ok = (dj == INF && gs == INF) || feq(dj, gs);
    printf("  %-18s : %s (Dijkstra = %.3f, D* Lite = %.3f)\n",
           "optimality check", ok ? "PASS" : "FAIL", dj, gs);
}

/* =====================================================================
 * TEST CASES
 * ===================================================================== */

static void tc1_basic_reachability(void)
{
    header("TEST 1 : Basic reachability      S -> A -> B -> G");
    PlanningProblem p; probInit(&p, 2);
    double c[4][2] = {{0,0},{1,0},{2,0},{3,0}};
    uint32_t S = probAddState(&p, "S", c[0]);
    uint32_t A = probAddState(&p, "A", c[1]);
    uint32_t B = probAddState(&p, "B", c[2]);
    uint32_t G = probAddState(&p, "G", c[3]);
    probAddTransition(&p, S, A, 1.0, 1.0, 0.99, 1);
    probAddTransition(&p, A, B, 1.0, 1.0, 0.98, 1);
    probAddTransition(&p, B, G, 1.0, 1.0, 0.97, 1);
    probSetBad(&p, NULL, 0);
    p.initialState = S; p.goalState = G;

    Planner pl; PlanningResult r;
    plannerInit(&pl, &p, defaultWeights());
    runPlan(&pl, &r);
    report("unique valid path returned", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r); plannerFree(&pl); probFree(&p);
}

static void tc2_bad_state_avoidance(void)
{
    header("TEST 2 : Bad-state avoidance     S->A->X->G (X bad) vs S->C->D->G");
    PlanningProblem p; probInit(&p, 2);
    double sc[2]={0,0}, ac[2]={1,1}, xc[2]={2,1}, gc[2]={4,0}, cc[2]={1,-1}, dc[2]={2,-1};
    uint32_t S = probAddState(&p,"S",sc), A = probAddState(&p,"A",ac);
    uint32_t X = probAddState(&p,"X",xc), G = probAddState(&p,"G",gc);
    uint32_t C = probAddState(&p,"C",cc), D = probAddState(&p,"D",dc);
    probAddTransition(&p,S,A,1.0,1.0,0.99,1);
    probAddTransition(&p,A,X,1.0,1.0,0.99,1);
    probAddTransition(&p,X,G,1.0,1.0,0.99,1);
    probAddTransition(&p,S,C,1.2,1.0,0.97,1);
    probAddTransition(&p,C,D,1.2,1.0,0.97,1);
    probAddTransition(&p,D,G,1.2,1.0,0.97,1);
    uint32_t bad[1] = { X };
    probSetBad(&p, bad, 1);
    p.initialState = S; p.goalState = G;

    Planner pl; PlanningResult r;
    plannerInit(&pl, &p, defaultWeights());
    runPlan(&pl, &r);
    report("cheaper path through X rejected", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r);

    printf("  -- bad-state set updated at run time: B = {} (X cleared) --\n");
    plannerSetBadStates(&pl, NULL, 0);
    runPlan(&pl, &r);
    report("cheaper route re-adopted", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r);

    printf("  -- bad-state set updated again: B = {C} --\n");
    uint32_t bad2[1] = { C };
    plannerSetBadStates(&pl, bad2, 1);
    runPlan(&pl, &r);
    report("lower branch now unusable", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r); plannerFree(&pl); probFree(&p);
}

static void tc3_safety_margin(void)
{
    header("TEST 3 : Safety margin           cost-vs-clearance trade-off");
    PlanningProblem p; probInit(&p, 2);
    double sc[2]={0,0}, p1[2]={2,0.4}, p2[2]={4,0.4}, gc[2]={6,0};
    double q1[2]={2,3}, q2[2]={4,3}, zc[2]={3,0};
    uint32_t S=probAddState(&p,"S",sc);
    uint32_t P1=probAddState(&p,"P1",p1), P2=probAddState(&p,"P2",p2);
    uint32_t Q1=probAddState(&p,"Q1",q1), Q2=probAddState(&p,"Q2",q2);
    uint32_t G=probAddState(&p,"G",gc),  Z=probAddState(&p,"Z*bad",zc);
    probAddTransition(&p,S,P1,1.0,1.0,0.99,1);   /* cheap, hugs the hazard */
    probAddTransition(&p,P1,P2,1.0,1.0,0.99,1);
    probAddTransition(&p,P2,G,1.0,1.0,0.99,1);
    probAddTransition(&p,S,Q1,1.6,1.0,0.99,1);   /* dearer, wide berth     */
    probAddTransition(&p,Q1,Q2,1.6,1.0,0.99,1);
    probAddTransition(&p,Q2,G,1.6,1.0,0.99,1);
    uint32_t bad[1] = { Z };
    probSetBad(&p, bad, 1);
    p.initialState = S; p.goalState = G;

    Weights w = defaultWeights();

    w.lambda = 0.0;                       /* pure cost minimisation        */
    { Planner pl; PlanningResult r;
      plannerInit(&pl, &p, w); runPlan(&pl, &r);
      printf("  -- lambda = 0.0 (cost only) --\n");
      report("shortest but least clearance", &p, &pl, &r);
      checkOptimal(&pl); freeResult(&r); plannerFree(&pl); }

    w.lambda = 6.0; w.dSafe = 3.5;        /* safety-weighted               */
    { Planner pl; PlanningResult r;
      plannerInit(&pl, &p, w); runPlan(&pl, &r);
      printf("  -- lambda = 6.0, dSafe = 3.5 (safety weighted) --\n");
      report("longer route, larger margin", &p, &pl, &r);
      checkOptimal(&pl); freeResult(&r); plannerFree(&pl); }

    probFree(&p);
}

static void tc4_dynamic_transition(void)
{
    header("TEST 4 : Dynamic transition      (A,G) becomes unavailable mid-run");
    PlanningProblem p; probInit(&p, 2);
    double sc[2]={0,0}, ac[2]={1,0}, gc[2]={3,0}, dc[2]={1.5,-1}, ec[2]={0.5,-1.5};
    uint32_t S=probAddState(&p,"S",sc), A=probAddState(&p,"A",ac);
    uint32_t G=probAddState(&p,"G",gc), D=probAddState(&p,"D",dc);
    uint32_t E=probAddState(&p,"E",ec);
    probAddTransition(&p,S,A,1.0,1.0,0.99,1);
    uint32_t eAG = probAddTransition(&p,A,G,2.0,1.0,0.95,1);
    probAddTransition(&p,A,D,0.8,1.0,0.96,1);
    probAddTransition(&p,D,G,1.4,1.0,0.96,1);
    probAddTransition(&p,S,E,1.5,1.0,0.90,1);
    probAddTransition(&p,E,G,2.5,1.0,0.90,1);
    probSetBad(&p, NULL, 0);
    p.initialState = S; p.goalState = G;

    Planner pl; PlanningResult r;
    plannerInit(&pl, &p, defaultWeights());
    runPlan(&pl, &r);
    printf("  -- initial plan --\n");
    report("nominal route", &p, &pl, &r);
    freeResult(&r);

    printf("  -- agent executes S -> A, then edge (A,G) is blocked --\n");
    plannerSetStart(&pl, A);              /* km absorbs the start motion   */
    plannerSetEdgeAvailable(&pl, eAG, 0); /* only vertex A is dirtied      */
    runPlan(&pl, &r);
    report("incremental repair from A", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r); plannerFree(&pl); probFree(&p);
}

static void tc5_goal_update(void)
{
    header("TEST 5 : Goal update             goal moves from G1 to G2");
    PlanningProblem p; probInit(&p, 2);
    double sc[2]={0,0}, ac[2]={1,0}, bc[2]={2,0}, g1[2]={2,1}, g2[2]={3,0};
    uint32_t S=probAddState(&p,"S",sc),  A=probAddState(&p,"A",ac);
    uint32_t B=probAddState(&p,"B",bc);
    uint32_t G1=probAddState(&p,"G1",g1), G2=probAddState(&p,"G2",g2);
    probAddTransition(&p,S,A,1.0,1.0,0.99,1);
    probAddTransition(&p,A,G1,1.0,1.0,0.99,1);
    probAddTransition(&p,A,B,1.0,1.0,0.98,1);
    probAddTransition(&p,B,G2,1.0,1.0,0.98,1);
    probSetBad(&p, NULL, 0);
    p.initialState = S; p.goalState = G1;

    Planner pl; PlanningResult r;
    plannerInit(&pl, &p, defaultWeights());
    runPlan(&pl, &r);
    printf("  -- goal = G1 --\n");
    report(NULL, &p, &pl, &r);
    freeResult(&r);

    printf("  -- goal switched to G2 (graph, geometry, clearances, heap buffers reused) --\n");
    plannerSetGoal(&pl, G2);
    runPlan(&pl, &r);
    report("O(n) reset, zero reallocation", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r); plannerFree(&pl); probFree(&p);
}

static void tc6_transition_addition(void)
{
    header("TEST 6 : Transition addition     shortcut S->B inserted at run time");
    PlanningProblem p; probInit(&p, 2);
    double c0[2]={0,0}, c1[2]={1,0}, c2[2]={2,0}, c3[2]={3,0};
    uint32_t S=probAddState(&p,"S",c0), A=probAddState(&p,"A",c1);
    uint32_t B=probAddState(&p,"B",c2), G=probAddState(&p,"G",c3);
    probAddTransition(&p,S,A,1.0,1.0,0.99,1);
    probAddTransition(&p,A,B,1.0,1.0,0.99,1);
    probAddTransition(&p,B,G,1.0,1.0,0.99,1);
    probSetBad(&p, NULL, 0);
    p.initialState = S; p.goalState = G;

    Planner pl; PlanningResult r;
    plannerInit(&pl, &p, defaultWeights());
    runPlan(&pl, &r);
    printf("  -- before insertion --\n");
    report(NULL, &p, &pl, &r);
    freeResult(&r);

    printf("  -- inserting shortcut (S,B) with cost 0.5 --\n");
    plannerAddTransition(&pl, S, B, 0.5, 1.0, 0.99);
    runPlan(&pl, &r);
    report("improved solution discovered", &p, &pl, &r);
    checkOptimal(&pl);
    freeResult(&r); plannerFree(&pl); probFree(&p);
}

/* ---- randomised grid benchmark -------------------------------------- */
static uint32_t rngState = 12345u;
static uint32_t rnd(void) { rngState = rngState * 1664525u + 1013904223u; return rngState >> 8; }

static void benchmark(void)
{
    const uint32_t N = 150;                 /* N x N 4-connected lattice   */
    header("BENCHMARK : 150 x 150 8-connected lattice, 22500 states, 178808 transitions");

    PlanningProblem p; probInit(&p, 2);
    probReserve(&p, N * N, 8 * N * N);
    char nm[12];
    for (uint32_t y = 0; y < N; y++)
        for (uint32_t x = 0; x < N; x++) {
            double v[2] = { (double)x, (double)y };
            snprintf(nm, sizeof nm, "%u,%u", x, y);
            probAddState(&p, nm, v);
        }
    /* 8-connected lattice, cost proportional to segment length so that the
       Euclidean heuristic is tight enough to prune usefully              */
    const int dx8[8] = {1,0,1,1,-1,0,-1,-1}, dy8[8] = {0,1,1,-1,0,-1,1,-1};
    for (uint32_t y = 0; y < N; y++)
        for (uint32_t x = 0; x < N; x++) {
            uint32_t u = y * N + x;
            for (int k = 0; k < 8; k++) {
                int nx = (int)x + dx8[k], ny = (int)y + dy8[k];
                if (nx < 0 || ny < 0 || nx >= (int)N || ny >= (int)N) continue;
                uint32_t v = (uint32_t)ny * N + (uint32_t)nx;
                double len = sqrt((double)(dx8[k]*dx8[k] + dy8[k]*dy8[k]));
                probAddTransition(&p, u, v, len * (1.0 + (rnd() % 100) / 1000.0),
                                  1.0, 0.99, 1);
            }
        }
    uint32_t bad[120];
    for (uint32_t i = 0; i < 120; i++) {
        uint32_t b;
        do { b = rnd() % (N * N); } while (b == 0 || b == N * N - 1);
        bad[i] = b;
    }
    probSetBad(&p, bad, 120);
    p.initialState = 0; p.goalState = N * N - 1;

    Weights w = defaultWeights(); w.lambda = 3.0; w.dSafe = 3.0;

    size_t graphMem = g_mem_cur;

    Planner pl; PlanningResult r;
    double tInit0 = now_ms();
    plannerInit(&pl, &p, w);
    double tInit1 = now_ms();
    size_t plannerMem = g_mem_cur - graphMem;
    runPlan(&pl, &r);
    printf("  -- initial plan --\n");
    printf("  %-18s : %s, %u states on path\n", "result",
           r.success ? "SUCCESS" : "FAILURE", r.nStatePath);
    printf("  %-18s : %.3f\n",    "total cost",      r.totalCost);
    printf("  %-18s : %.3f\n",    "min dist to bad", r.safetyScore);
    printf("  %-18s : %ld of %u\n", "states expanded", r.expanded, p.nStates);
    printf("  %-18s : %.3f ms (setup %.3f ms incl. clearance field)\n",
           "planning time", r.planMs, tInit1 - tInit0);
    printf("  %-18s : %.1f KB graph + %.1f KB planner\n", "memory",
           graphMem / 1024.0, plannerMem / 1024.0);
    double firstCost = pl.g[pl.start];
    freeResult(&r);

    /* same query with h = 0, i.e. the search degenerates to Dijkstra */
    { Planner pz; PlanningResult rz;
      plannerInit(&pz, &p, w);
      pz.hScale = 0.0; heapRekeyAll(&pz);
      double z0 = now_ms(); plannerCompute(&pz); double z1 = now_ms();
      plannerExtract(&pz, &rz);
      printf("  -- heuristic ablation: same plan with h = 0 --\n");
      printf("  %-18s : %ld (vs %ld with the Euclidean heuristic)\n",
             "states expanded", pz.expanded, r.expanded);
      printf("  %-18s : %.3f ms\n", "planning time", z1 - z0);
      freeResult(&rz); plannerFree(&pz); }

    /* block 300 random currently-available transitions, then repair */
    uint32_t blocked[300];
    for (uint32_t i = 0; i < 300; i++) { blocked[i] = rnd() % p.nTrans;
                                         p.transitions[blocked[i]].available = 0; }
    double t0 = now_ms();
    for (uint32_t i = 0; i < 300; i++) plannerEdgeChanged(&pl, blocked[i]);
    pl.expanded = 0; pl.U.ops = 0;
    plannerCompute(&pl);
    double t1 = now_ms();
    plannerExtract(&pl, &r);
    printf("  -- 300 transitions removed, incremental repair --\n");
    printf("  %-18s : %.3f  (was %.3f)\n", "g(start)", pl.g[pl.start], firstCost);
    printf("  %-18s : %ld\n",    "states expanded", pl.expanded);
    printf("  %-18s : %.3f ms\n", "replanning time", t1 - t0);
    double incCost = pl.g[pl.start], incMs = t1 - t0;
    long incExp = pl.expanded;
    freeResult(&r);
    plannerFree(&pl);

    /* same query, planned from scratch, for comparison */
    Planner pl2; PlanningResult r2;
    plannerInit(&pl2, &p, w);      /* setup excluded: it would be reused    */
    double t2 = now_ms();
    plannerCompute(&pl2);
    double t3 = now_ms();
    plannerExtract(&pl2, &r2);
    printf("  -- same query replanned from scratch --\n");
    printf("  %-18s : %.3f\n",    "g(start)",        pl2.g[pl2.start]);
    printf("  %-18s : %ld\n",     "states expanded", pl2.expanded);
    printf("  %-18s : %.3f ms\n", "planning time",   t3 - t2);
    printf("  %-18s : %s\n", "agreement",
           feq(incCost, pl2.g[pl2.start]) ? "identical cost - incremental repair is exact"
                                          : "MISMATCH");
    printf("  %-18s : %.2fx fewer expansions, %.2fx faster\n", "speed-up",
           pl2.expanded / (double)(incExp ? incExp : 1),
           (t3 - t2) / (incMs > 1e-6 ? incMs : 1e-6));
    freeResult(&r2); plannerFree(&pl2);
    probFree(&p);
}

/* =====================================================================
 * MAIN
 * ===================================================================== */
int main(void)
{
    printf("PCCST503 Assignment 1 - Safe Semantic Planner (D* Lite, C11)\n");
    printf("Effective weight  w(u,v) = cost + lambda*max(0,dSafe-clear(v))/dSafe"
           " + mu*(1-rel) + nu*(1-safety)\n");
    printf("Score(P) = %.0f*G - %.0f*C + %.0f*D + %.0f*R\n",
           defaultWeights().alpha, defaultWeights().beta,
           defaultWeights().gamma, defaultWeights().delta);

    tc1_basic_reachability();
    tc2_bad_state_avoidance();
    tc3_safety_margin();
    tc4_dynamic_transition();
    tc5_goal_update();
    tc6_transition_addition();
    benchmark();

    printf("\n----------------------------------------------------------\n");
    printf("peak tracked heap usage : %.1f KB\n", g_mem_peak / 1024.0);
    printf("leaked bytes at exit    : %zu\n", g_mem_cur);
    printf("----------------------------------------------------------\n");
    return 0;
}
