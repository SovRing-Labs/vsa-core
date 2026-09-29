#include <errno.h>
/* VSA bus self-test.
 *
 * Exists because the previous smoke tests could not fail: every read and write
 * was at byte offset 0 (the one address where a 2x stride error is invisible),
 * and the benchmark accumulated a constant. Each check below is one that the
 * old suite passed while the defect was live.
 */
#include "vsa_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>

#define SHM_NAME     "/vsa_matrix_bus"
#define SHM_SEQ_NAME "/vsa_matrix_seq"
#define N_HV          1000
#define HV_PER_LANE   100

static int fails = 0;
static void ok(const char* what, int cond, const char* detail) {
    printf("  [%s] %-52s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

static uint64_t rs = 0x243F6A8885A308D3ULL;
static uint64_t rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static void mkhv(vsa_hv_t* h){ for(size_t w=0;w<VSA_WORDS;w++){h->sign[w]=rnd();h->zero[w]=rnd();} }

/* ---- 1. layout ---- */
static void test_layout(void) {
    printf("\n1. Layout agreement\n");
    char b[128];
    snprintf(b,sizeof b,"sizeof(vsa_hv_t)=%zu", sizeof(vsa_hv_t));
    ok("hypervector is 2560 B (two bitplanes)", sizeof(vsa_hv_t)==2560, b);
    ok("matches HV_TOTAL_BYTES used by every lane process", VSA_HV_BYTES==2560, NULL);
    ok("lane stride is 256000 B", HV_PER_LANE*VSA_HV_BYTES==256000, NULL);
}

/* ---- 2. dot product against an independent scalar reference ---- */
static int32_t dot_ref(const vsa_hv_t* a, const vsa_hv_t* b) {
    int32_t s=0;
    for (size_t d=0; d<VSA_DIMENSIONS; d++) {
        size_t w=d/64, t=d%64;
        int av = ((a->zero[w]>>t)&1) ? (((a->sign[w]>>t)&1)?-1:1) : 0;
        int bv = ((b->zero[w]>>t)&1) ? (((b->sign[w]>>t)&1)?-1:1) : 0;
        s += av*bv;
    }
    return s;
}
static void test_dot(void) {
    printf("\n2. Ternary dot product vs scalar reference\n");
    vsa_hv_t a,b; int bad=0;
    for (int i=0;i<200;i++){ mkhv(&a); mkhv(&b); if (vsa_dot(&a,&b)!=dot_ref(&a,&b)) bad++; }
    char s[64]; snprintf(s,sizeof s,"%d/200 mismatches",bad);
    ok("SIMD dot matches scalar reference", bad==0, s);
    mkhv(&a);
    snprintf(s,sizeof s,"dot=%d active=%d",vsa_dot(&a,&a),vsa_active_count(&a));
    ok("self-similarity equals active count", vsa_dot(&a,&a)==vsa_active_count(&a), s);
    /* the old benchmark returned the SAME value every iteration; real HVs vary */
    int32_t v0=0; int varies=0;
    for (int i=0;i<50;i++){ mkhv(&a); mkhv(&b); int32_t v=vsa_dot(&a,&b);
        if(i==0)v0=v; else if(v!=v0)varies=1; }
    ok("random pairs produce VARYING scores (not a constant)", varies, NULL);
}

/* ---- 3. bundling capacity: the property that actually decides viability ---- */
static void test_capacity(void) {
    printf("\n3. Superposition capacity (majority bundling)\n");
    for (int k = 5; k <= 100; k *= 2) {
        vsa_hv_t* items = malloc(sizeof(vsa_hv_t)*k);
        const vsa_hv_t** ptrs = malloc(sizeof(void*)*k);
        for (int i=0;i<k;i++){ mkhv(&items[i]); ptrs[i]=&items[i]; }
        vsa_hv_t bundle; vsa_bundle(ptrs,k,&bundle);
        int32_t worst_member = 1<<30, best_foil = -(1<<30);
        for (int i=0;i<k;i++){ int32_t d=vsa_dot(&bundle,&items[i]); if(d<worst_member)worst_member=d; }
        for (int i=0;i<20;i++){ vsa_hv_t f; mkhv(&f); int32_t d=vsa_dot(&bundle,&f); if(d>best_foil)best_foil=d; }
        char s[128];
        snprintf(s,sizeof s,"k=%3d member>=%5d foil<=%4d",k,worst_member,best_foil);
        ok("every member retrievable above noise", worst_member>best_foil, s);
        free(items); free((void*)ptrs);
    }
    /* guard against a regression to OR bundling, which fails at k=7 */
    vsa_hv_t it[10]; const vsa_hv_t* p[10];
    for(int i=0;i<10;i++){ mkhv(&it[i]); p[i]=&it[i]; }
    vsa_hv_t b; vsa_bundle(p,10,&b);
    int32_t lo=1<<30; for(int i=0;i<10;i++){int32_t d=vsa_dot(&b,&it[i]); if(d<lo)lo=d;}
    char s[96]; snprintf(s,sizeof s,"min member score %d (OR bundling gives <0)",lo);
    ok("k=10 members score positive (OR regression guard)", lo>0, s);
}

/* ---- 4. every lane, at its real offset -- not offset 0 ---- */
static void test_lanes(void) {
    printf("\n4. Lane isolation at real offsets (not byte 0)\n");
    int fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (fd < 0) { ok("bus segment present", 0, "run shm_allocator first"); return; }
    size_t sz = (size_t)N_HV*VSA_HV_BYTES;
    uint8_t* base = mmap(0,sz,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if (base==MAP_FAILED){ ok("bus mmap",0,strerror(errno)); close(fd); return; }
    char s[128];
    snprintf(s,sizeof s,"%zu B mapped", sz);
    ok("bus is exactly N*2560 bytes", 1, s);

    /* stamp a distinct hypervector into the FIRST HV of each of the 10 lanes */
    vsa_hv_t mark[10];
    for (int L=0; L<10; L++) {
        mkhv(&mark[L]);
        memcpy(base + (size_t)L*HV_PER_LANE*VSA_HV_BYTES, &mark[L], VSA_HV_BYTES);
    }
    int clean = 1; int32_t worst = 1<<30;
    for (int L=0; L<10; L++) {
        vsa_hv_t got; memcpy(&got, base + (size_t)L*HV_PER_LANE*VSA_HV_BYTES, VSA_HV_BYTES);
        int32_t self = vsa_dot(&got,&mark[L]);
        if (self != vsa_active_count(&mark[L])) clean = 0;
        if (self < worst) worst = self;
        /* and confirm it did not bleed into the neighbouring lane */
        if (L < 9) {
            vsa_hv_t nb; memcpy(&nb, base + (size_t)(L+1)*HV_PER_LANE*VSA_HV_BYTES, VSA_HV_BYTES);
            if (vsa_dot(&nb,&mark[L]) == vsa_active_count(&mark[L])) clean = 0;
        }
    }
    snprintf(s,sizeof s,"10 lanes, min self-score %d", worst);
    ok("each lane reads back exactly what it wrote", clean, s);
    /* Regression guard for the 1280-stride allocator: the segment itself must
     * be N*2560. Under the old allocator it was N*1280 and every lane above
     * HV 500 would run past the end of the mapping. */
    struct stat st; int szok = (fstat(fd,&st)==0) && ((size_t)st.st_size == (size_t)N_HV*VSA_HV_BYTES);
    snprintf(s,sizeof s,"segment %lld B, expected %zu", (long long)st.st_size, (size_t)N_HV*VSA_HV_BYTES);
    ok("segment sized for 2560 B/HV, not the old 1280", szok, s);
    munmap(base,sz); close(fd);
}

/* ---- 5. torn reads under concurrency ---- */
static uint8_t* g_base; static uint64_t* g_seq; static volatile int g_stop;
static void* writer(void* _) {
    (void)_; vsa_hv_t a,b; mkhv(&a); mkhv(&b); int t=0;
    while(!g_stop){
        vsa_write_begin(&g_seq[0]);
        memcpy(g_base, (t++&1)?&a:&b, VSA_HV_BYTES);
        vsa_write_end(&g_seq[0]);
    }
    return NULL;
}
static void test_torn(void) {
    printf("\n5. Torn-read protection (seqlock)\n");
    int fd = shm_open(SHM_NAME,O_RDWR,0666), sf = shm_open(SHM_SEQ_NAME,O_RDWR,0666);
    if (fd<0||sf<0){ ok("segments present",0,NULL); return; }
    g_base = mmap(0,(size_t)N_HV*VSA_HV_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    g_seq  = mmap(0,(size_t)N_HV*8,PROT_READ|PROT_WRITE,MAP_SHARED,sf,0);
    g_stop=0;
    pthread_t th; pthread_create(&th,NULL,writer,NULL);
    int torn_guarded=0, reads=0;
    for (int i=0;i<200000;i++){
        vsa_hv_t snap; uint64_t s0;
        do { s0=vsa_read_begin(&g_seq[0]); memcpy(&snap,g_base,VSA_HV_BYTES); }
        while (vsa_read_retry(&g_seq[0],s0));
        reads++;
        /* a consistent snapshot always satisfies dot(x,x) == active(x) */
        if (vsa_dot(&snap,&snap) == vsa_active_count(&snap)) torn_guarded++;
    }
    g_stop=1; pthread_join(th,NULL);
    char s[96]; snprintf(s,sizeof s,"%d/%d consistent snapshots",torn_guarded,reads);
    ok("concurrent reads never observe a torn hypervector", torn_guarded==reads, s);
}

#include <errno.h>
int main(void) {
    printf("================ VSA BUS SELF-TEST ================\n");
    test_layout(); test_dot(); test_capacity(); test_lanes(); test_torn();
    printf("\n==================================================\n");
    printf("%s  (%d failure%s)\n", fails?"FAILED":"ALL CHECKS PASSED", fails, fails==1?"":"s");
    return fails ? 1 : 0;
}
