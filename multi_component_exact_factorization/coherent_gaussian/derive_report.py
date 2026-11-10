"""Generate the requested Korean derivation/validation PDF, using actual receipts.

Optional report-only dependency: reportlab. It is not imported by propagators.
Equations are rendered with matplotlib mathtext; no LaTeX installation needed.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np


def collect(roots):
    rows=[]
    for root in roots:
        for p in sorted(Path(root).rglob('summary.json')):
            s=json.loads(p.read_text());cp=p.parent/'config.json'
            if not cp.exists():continue
            c=json.loads(cp.read_text())
            if 'representation' not in c:continue
            m=json.loads((p.parent/'metrics.json').read_text())
            s.update(path=str(p.parent),model=c['model'],basis=c['representation'],K=c['K'],
                regularization=c.get('shape_regularization','not recorded'),constraint=c['energy_constraint'],
                nf=c['nf_reference'],rtol=c['rtol'],max_step=c['max_step_au'])
            if m:
                s['photon_error']=max(abs(x['photon_number']-x['reference_photon_number']) for x in m) if 'photon_number' in m[0] else None
                s['population_error']=max(abs(x['excited_population']-x['reference_excited_population']) for x in m) if 'excited_population' in m[0] else None
                s['max_density_l1']=max(x['density_l1'] for x in m)
                s['matched_initial_l2']=max(x['matched_initial_l2'] for x in m)
                s['local_potential_rms_Ha']=max(x.get('local_potential_rms_Ha',0) for x in m)
                s['outside_fit_mass']=max(x.get('outside_local_fit_mass',0) for x in m)
            audit=p.parent/'quality_audit.json'
            if audit.exists():s['quality_audit']=json.loads(audit.read_text())
            rows.append(s)
    return rows


def report(out,roots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from xml.sax.saxutils import escape
    pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
    font='HYSMyeongJo-Medium'
    textstyle=ParagraphStyle('body',fontName=font,fontSize=10,leading=16,wordWrap='CJK',spaceAfter=8)
    head=ParagraphStyle('heading',parent=textstyle,fontSize=15,leading=21,spaceBefore=10,spaceAfter=12)
    small=ParagraphStyle('small',parent=textstyle,fontSize=8,leading=11)
    story=[];buffers=[]
    def para(s):story.append(Paragraph(s,textstyle))
    def title(s):story.append(Paragraph(s,head))
    def eq(s):
        b=io.BytesIO();fig=plt.figure(figsize=(8,.65))
        fig.text(.02,.5,'$'+s+'$',fontsize=16,va='center');fig.savefig(b,format='png',dpi=180,bbox_inches='tight',pad_inches=.08)
        plt.close(fig);b.seek(0);buffers.append(b)
        im=Image(b);scale=min(460/im.imageWidth,72/180);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
        story.extend([im,Spacer(1,8)])
    def page():story.append(PageBreak())
    def table(data,widths):
        cells=[[Paragraph(escape(str(x)),small) for x in row] for row in data]
        tab=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.25,colors.lightgrey),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6)]));story.append(tab)
    title('고정 폭 복소 Gaussian 전파: 유도, 구현, 검증')
    para('작성일: 2026-11-10. 기존 positive-bridge Gaussian 전파를 대체하지 않는 독립 실험 모듈이다. '
         '이 문서는 실행한 수치 시험과 아직 해결하지 못한 문제를 구분한다. 좋은 에너지 보존은 좋은 파동함수 근사의 충분조건이 아니다.')
    para('핵심 변경은 양의 bridge 진폭만 Gaussian 밀도 혼합으로 제한하던 방법에서, 복소 전자-내부좌표 상태 G 자체를 공통 coherent Gaussian 집합으로 전개하는 것이다. '
         '조건부 전자계수 C(q)는 상수로 고정하지 않고 전파된 G/Λ에서 복원한다. 전파 변수에 전역 q 격자는 없다.')
    table([['계산','이번 구현 범위'],['PEN 2026 benchmark','고정 R BO/CBO + 움직이는 R의 저장소 CTMQC EL/POL/실험적 Gaussian-POL 비교'],
           ['Real-electron PEN','기존 frozen-heavy 분자 potential과 실공간 전자 Hamiltonian을 유지한 고정 R 시험. 모든 nx 상태 유지'],
           ['MCEF','q=proton, R=heavy. 고정 heavy의 국소 polynomial surrogate만 검증. 원래 전체 MCEF 성공을 뜻하지 않음'],
           ['Native reference','저장소 Python CTMQC. 저자 Fortran G-CTMQC를 이번에 재실행한 것은 아님']], [110,365])
    title('1. 표기와 상태공간')
    eq(r'G(x,q;R,t)=\sum_{k=1}^{N_e}\sum_{j=1}^{K}d_{kj}(t)\phi_k(x;q,R)g_j(q;t)')
    para('R은 이 유도에서는 고정된 매개변수다. 움직이는 R의 diabatic 구현은 뒤의 별도 절에서 설명한다. '
         '전자기저는 각 q에서 정규직교한다. 일반 PEN의 BO는 R에만, CBO는 q와 R 모두에 의존한다. '
         '원래 MCEF에는 photon이 없으므로 그 전자기저는 BO이며 CBO라고 부르지 않는다.')
    eq(r'\Lambda(q)=\sqrt{\sum_k |G_k(q)|^2},\qquad C_k(q)=G_k(q)/\Lambda(q)')
    para('q-dependent 전자기저에서는 G_k는 해당 기저 계수이며, 실제 전자 벡터는 φG다. 정규직교성으로 Λ는 동일하다. '
         'C의 q 의존성은 복소 Gaussian 간 간섭과 서로 다른 전자계수에서 생긴다. 유한 K는 여전히 표현력 근사다.')
    page();title('2. 고정 폭 coherent Gaussian과 시간미분')
    eq(r'g_j=(2\pi\sigma^2)^{-1/4}\exp[-(q-Q_j)^2/(4\sigma^2)+iP_j(q-Q_j/2)]')
    eq(r'z_j=Q_j/(2\sigma)+i\sigma P_j,\qquad \sigma=(2m_q\omega)^{-1/2}')
    para('σ는 밀도 표준편차다. photon에서는 vacuum 폭이며 고정한다. 각 Gaussian의 Q와 P는 움직인다. '
         '폭이 고정되어도 여러 Gaussian의 간섭으로 전체 밀도 폭과 모양은 달라진다. MCEF proton 시험은 기존 초기 폭을 σ로 사용하며 이를 photon vacuum이라고 부르지 않는다.')
    eq(r'|g(z)\rangle=e^{-|z|^2/2}e^{z\hat a^\dagger}|0\rangle')
    eq(r'|\dot g_j\rangle=\dot z_j\hat a^\dagger|g_j\rangle-\mathrm{Re}(z_j^*\dot z_j)|g_j\rangle')
    para('증명: 두 지수의 곱을 직접 미분하면 첫 번째 지수에서 −Re(z* zdot), 두 번째 지수에서 zdot a†가 나온다. '
         '따라서 독립적인 phase reset이나 유한차분 q 미분은 필요 없다. 각 Gaussian의 추가 상수 위상은 복소 d에 포함된다.')
    eq(r'u_{kj}=\dot d_{kj}-d_{kj}\mathrm{Re}(z_j^*\dot z_j)')
    eq(r'|\dot G\rangle=\sum_{kj}u_{kj}|\phi_k g_j\rangle+\sum_j\dot z_j\sum_k d_{kj}|\phi_k\hat a^\dagger g_j\rangle')
    para('u는 실제 d의 시간미분과 다르다. 선형문제를 풀어 u, zdot을 구한 뒤 ddot=u+d Re(z* zdot)으로 환산한다. '
         'Qdot=2σ Re(zdot), Pdot=Im(zdot)/σ이다. 이것이 요청한 고정 폭 parameter 전파식이다.')
    page();title('3. 계수와 Gaussian parameter의 결합 행렬식')
    para('작용의 정지 조건 또는 복소 접공간 Dirac–Frenkel 조건을 사용한다. q 전체의 C를 따로 업데이트하는 기존 실수 진폭 projection과 다른 방법론이다.')
    eq(r'\langle B_\mu|i\dot G-HG\rangle=0,\qquad iM v=b')
    eq(r'B^c_{ki}=\phi_k g_i,\qquad B^z_i=\sum_k d_{ki}\phi_k\hat a^\dagger g_i')
    eq(r'M_{\mu\nu}=\langle B_\mu|B_\nu\rangle,\quad b_\mu=\langle B_\mu|HG\rangle,\quad v=(u,\dot z)')
    eq(r'S_{ij}=\exp[-(|z_i|^2+|z_j|^2)/2+z_i^*z_j]')
    eq(r'M^{cc}_{ki,lj}=\delta_{kl}S_{ij},\quad M^{cz}_{ki,j}=d_{kj}z_i^*S_{ij}')
    eq(r'M^{zz}_{ij}=\left(\sum_kd_{ki}^*d_{kj}\right)(1+z_i^*z_j)S_{ij}')
    para('증명: a|z>=z|z>, [a,a†]=1을 사용한다. <zi|a†|zj>=zi* Sij, '
         '<zi|a a†|zj>=(1+zi* zj)Sij이다. q-dependent 전자기저도 매 q에서 정규직교하므로 metric의 식은 동일하다. '
         'Hamiltonian 행렬에는 그 기저의 q 미분이 반드시 들어간다.')
    eq(r'b^c_{ki}=\sum_{lj}\langle\phi_k g_i|H|\phi_l g_j\rangle d_{lj}')
    eq(r'b^z_i=\sum_{klj}d_{ki}^*\langle\phi_k\hat a^\dagger g_i|H|\phi_l g_j\rangle d_{lj}')
    para('계수만 고정 기저에서 전파하는 경우에는 i S ddot=H d가 된다. Gaussian이 움직이면 overlap-time 연결항이 추가되며, 위 결합식이 그 항을 이미 포함한다.')
    title('3.1 Schur complement와 수치적으로 안전한 구성')
    eq(r'u_0=-i(M^{cc})^+b^c,\quad Z=M^{zz}-M^{zc}(M^{cc})^+M^{cz}')
    eq(r'Z\dot z=-ib^z-M^{zc}u_0,\quad u=u_0-(M^{cc})^+M^{cz}\dot z')
    para('초기 빈 Gaussian이나 서로 가까운 Gaussian에서는 rank가 떨어진다. 두 큰 행렬을 뺄 경우 상쇄 오차로 Z가 음수가 될 수 있어, '
         '구현은 coefficient span에 직교하는 실제 tangent residual의 Gram matrix로 Z를 만든다. '
         'coherent state의 해석적 Fock 계수를 일시적으로 QR/SVD에 사용하지만 Fock 계수를 전파하지 않는다. 그 보조 전개의 누락 norm도 기록한다.')
    page();title('4. 정규화, 에너지 보존, 잔차의 차이')
    eq(r'\dot G=-iP_T HG,\quad P_T=B(B^\dagger B)^+B^\dagger')
    para('복소 접공간의 직교 projection P_T는 Hermitian이다. G가 보존된 coefficient span에 속하면 P_T G=G이다. '
         '시간독립 Hermitian H에서 다음 두 식이 성립한다. 실제 계산에서는 rank cutoff, quadrature와 시간적분 오차를 별도로 측정해야 한다.')
    eq(r'\dot N=2\mathrm{Re}\langle G|-iP_T HG\rangle=0')
    eq(r'\dot E=2\mathrm{Re}[i\langle HG|P_T HG\rangle]/N=0')
    para('여기서 E=<G|H|G>/N이다. 에너지 보존의 증명은 P_T가 정확한 Schrödinger 시간미분을 재현한다는 뜻이 아니다. '
         '접공간에 없는 성분은 버리므로 state와 population 오차가 클 수 있다. K=1의 좋은 에너지 보존과 나쁜 상태 오차가 실제 반례다.')
    eq(r'\|G(t)-G_{ref}(t)\|\leq\|G(0)-G_{ref}(0)\|+\int_0^t\|\dot G+iHG\|\,ds')
    para('증명: 오차 방정식 edot=−iHe+r를 variation of constants로 적분한다. Hermitian H의 propagator는 norm을 보존하므로 삼각부등식으로 위 식을 얻는다. '
         '진단용 reference는 우변 계산이나 online fitting에 사용되지 않는다. 저장 시점의 residual 적분은 유한 표본 근사이며 엄밀한 연속시간 상계의 인증이 아니다.')
    title('4.1 명시적 shape regularization')
    eq(r'Z=U\,\mathrm{diag}(\lambda_j)U^\dagger,\quad Z^+\ \longrightarrow\ U\,\mathrm{diag}\left(\frac{\lambda_j}{\lambda_j^2+\tau^2}\right)U^\dagger')
    para('이 smooth filter는 거의 빈 Gaussian의 급격한 rank 변화를 완화하기 위한 공개된 근사다. 정확한 TDVP 역행렬과 같다고 주장하지 않는다. '
         'coefficient 방정식은 유지하고, shape residual Gram factorization에 Hermitian filter를 적용한다. 이상적인 coefficient span에서는 생성자의 Hermiticity와 G 포함 조건이 남는다. '
         '코드는 raw norm/energy rate, shape-equation defect와 cutoff를 기록한다. τ를 낮추는 것이 항상 더 빠르거나 안정적인 것은 아니다.')
    title('4.2 선택적 에너지 제약')
    para('고정 R 계산의 선택 옵션은 coefficient 접공간에서 norm/energy 시간변화율을 0으로 하는 최소 correction을 추가한다. '
         '결과 에너지를 초기값으로 덮어쓰거나 전파 뒤 renormalization하는 방식이 아니다. 복소 TDVP가 이미 보존성을 갖기 때문에 보통 correction은 수치 오차 수준이다.')
    eq(r'\delta\dot G=aG+bP_c(H-E)G')
    eq(r'2\mathrm{Re}\langle G|\delta\dot G\rangle=-\dot N,\quad 2\mathrm{Re}\langle(H-E)G|\delta\dot G\rangle=-N\dot E')
    para('실수 a,b의 2×2 선형문제를 rank-aware least squares로 푼다. 외부 driving 또는 CTMQC source에는 이 조건을 허용하지 않는다. '
         '움직이는 핵과 내부계는 에너지를 교환하므로 각 trajectory 내부 에너지를 고정하는 것은 잘못이다.')
    page();title('5. C와 Λ의 복원: 무엇을 잃고 무엇을 유지하는가')
    eq(r'\dot\Lambda=\mathrm{Re}\sum_k C_k^*\dot G_k,\quad \dot C_k=(\dot G_k-C_k\dot\Lambda)/\Lambda')
    eq(r'\partial_t\sum_k|C_k|^2=0,\qquad \dot G_k=\dot\Lambda C_k+\Lambda\dot C_k')
    para('증명은 |Λ|²=Σ|Gk|²의 미분과 몫의 미분이다. 반드시 실제 projected Gdot을 사용한다. '
         '원래 HG의 radial 변화율과 다른 projected Λdot을 혼합하면 이전에 조사한 normalization/projection 불일치가 다시 생긴다. '
         'Λ=0인 node에서는 C가 정의되지 않으므로 density support 밖의 C와 connection을 해석하지 않는다.')
    para('같은 K개 Gaussian을 전자 상태 모두가 공유하지만 d_kj는 서로 다르다. 따라서 C_k(q)는 일반적으로 q-dependent이고 전자–광자 얽힘도 가능하다. '
         '그러나 모든 grid 함수 C_k(q)를 정확히 나타낼 수 있는 것은 아니다. density fitting뿐 아니라 complex-state L2, density L1, kinetic energy, photon number와 electronic population의 K 수렴을 확인해야 한다.')
    title('6. BO와 CBO: basis 의존성과 kinetic coupling')
    eq(r'H=-\frac{1}{2m_q}\partial_q^2+V(q,R)')
    eq(r'\langle\phi_k g_i|T_q|\phi_lg_j\rangle=\frac{1}{2m_q}\int dq\,[\partial_q(\phi_k g_i)]^*\partial_q(\phi_l g_j)')
    eq(r'A_{kl}=\langle\phi_k|\partial_q\phi_l\rangle,\quad B_{kl}=\langle\partial_q\phi_k|\partial_q\phi_l\rangle')
    eq(r"T_{ki,lj}=\frac{1}{2m_q}\int dq\,[g_i'^*g_j'\delta_{kl}+g_i'^*g_jA_{kl}+g_i^*g_j'A^\dagger_{kl}+g_i^*g_jB_{kl}]")
    para('integration by parts의 경계항이 사라지는 정의역에서 얻는 weak kinetic form이다. CBO 에너지만 바꿔 넣고 A,B를 버리면 같은 Hamiltonian이 아니다. '
         '구현은 local Gaussian-pair quadrature로 네 항을 모두 계산한다. truncated 전자공간에서도 B는 outside-state derivative 성분을 포함한다.')
    eq(r'\langle\phi_l|\partial_q\phi_k\rangle=\frac{\langle\phi_l|\partial_q V|\phi_k\rangle}{E_k-E_l}\quad(l\ne k)')
    para('증명: Vφk=Ekφk를 q로 미분하고 φl로 투영한다. 비축퇴와 선택한 smooth gauge가 필요하다. '
         'degeneracy 또는 anchor chart 이탈은 숨기지 않고 실패로 기록한다. BO/CBO가 완전해도 유한 K에서 허용하는 복소 함수족은 다르므로 결과가 다를 수 있다. '
         'CBO의 초기 fitting 오차와 이후 전파 오차를 분리하려고 matched-initial reference도 함께 저장한다.')
    page();title('7. 실제 Hamiltonian과 q-grid 없는 적분')
    eq(r'H_{PEN}=h(R)+\omega(\hat a^\dagger\hat a+1/2)+\omega g\sigma\mu(R)(\hat a+\hat a^\dagger)+\frac{1}{2}g^2\mu^2(R)')
    para('g는 dipole-gauge 계수이며 Rabi splitting이 아니다. DSE는 선택한 run의 config대로 on/off다. '
         'BO/diabatic에서는 coherent state의 normal-ordered 행렬원소를 해석적으로 계산한다. CBO에서는 Gaussian pair마다 따라가는 Gauss–Hermite 점을 사용한다. '
         '이 적분점은 전파 변수도 저장된 파동함수 격자도 아니다. quadrature 수렴은 여전히 필요하며, grid-free가 cost-free를 뜻하지 않는다.')
    eq(r'H_{00,ikjl}=S_{ij}[h_{kl}+\omega(z_i^*z_j+1/2)\delta_{kl}+\omega g\sigma\mu_{kl}(z_i^*+z_j)+\frac{1}{2}g^2(\mu^2)_{kl}]')
    para('H10=<a†gi|H|gj>는 <gi|aH|gj>에서 aH=Ha+[a,H]를 사용한다. '
         '[a,H]=ωa+ωgσμ이므로 H10=zj H00+S(ωzj I+ωgσμ)이다. 고정 R에서만 편의상 상수 에너지 shift를 사용하며 reference에도 동일하게 적용한다.')
    title('7.1 Real-electron PEN과 원래 MCEF의 구별')
    para('RealPen은 기존 frozen_potential과 DST-I 전자 kinetic을 그대로 가져오고 nx 전자 성분을 모두 유지한다. '
         '고정 R의 bare S1−S0 gap으로 vacuum 폭과 ω를 정할 수 있다. 기본 자동 g는 2-level 추정 splitting/ω=0.1에 해당하는 사전 설정일 뿐, 실제 splitting 인증은 아니다. '
         '로컬의 nx=17 시험은 production nx=359 격자 수렴을 대체하지 않는다.')
    para('MCEF에는 unsmoothed 1/|q+9.5| proton–fixed-ion pole이 있다. 그 점에서 0이 아닌 무한 지지 Gaussian과의 적분은 발산한다. '
         '이를 임의로 smoothing하거나 density mask로 없애지 않았다. 대신 q=−4 주변 반경 3의 pole-free 구간에서 분자 potential을 degree-6 polynomial로 근사한 별도 local surrogate를 사용한다. '
         'fit 영역 밖 확률과 실제 potential 대비 density-weighted RMS를 기록한다. 원래 MCEF의 장시간 검증을 위해서는 경계를 만족하는 Gaussian 형태 또는 명시적 domain 처방이 추가로 필요하다.')
    page();title('8. 움직이는 핵: CTMQC와의 연결 및 제한')
    para('저장소의 native POL CTMQC는 R,P와 전자×Fock 계수를 전파하고, EL은 R,q를 고전좌표로 전파한다. '
         '이번 실험적 Gaussian-POL은 기존 POL의 force, quantum momentum, integrated-force 방정식에서 source를 얻어 Gaussian 접공간에 project한다. '
         '새 force나 photon hopping rule을 임의로 만들지 않았다. 전자 basis는 고정 diabatic이므로 R-dependent basis 연결항을 누락하지 않는다.')
    eq(r'F_{CT}=\dot G_{POL}+iH_{POL}G,\qquad Mv=-ib+\langle B|F_{CT}\rangle')
    para('CT closure 평가 시 Gaussian을 유한 Fock subspace로 일시 투영한다. tail norm이 허용치를 넘으면 중단하고 nf 증가를 요구한다. '
         '따라서 q-grid는 없지만 polaritonic diagonalization 비용은 남는다. 이는 속도 최적화된 최종 이론이 아니라, 기존 source와 비교하기 위한 공개된 projection approximation이다.')
    para('Native는 fixed RK4, Gaussian은 adaptive DOP853이다. 전자상태는 projector overlap으로 tracking하며 inactive force histories를 reset한다. '
         'Native의 reset은 고정 timestep마다, Gaussian은 accepted adaptive step마다 이루어진다. reset 시점 차이도 수렴검사 대상이며 완전히 같은 이산 알고리즘이라고 주장하지 않는다. '
         'Native coefficient renormalization을 Gaussian에는 적용하지 않는다.')
    para('EL의 photon number는 vacuum Wigner 표본의 quadrature energy에서 1/2를 뺀 estimator다. 유한 표본에서 음수일 수 있으며 양자 Fock population과 같지 않다. '
         'QD/EL/POL은 bare S1×vacuum의 공통 preparation으로 비교하되, 논문의 upper-CBO EL preparation과는 구분한다. '
         'Ntraj가 작으면 sampling 오차와 trajectory closure 오차가 Gaussian 오차보다 클 수 있다.')
    para('R-dependent BO/CBO로 움직이는 핵까지 확장하면 basis의 물질시간미분과 R derivative coupling이 추가로 필요하다. '
         '고정 R의 BO/CBO 시험만으로 이 전체 closure를 구현했다고 주장하지 않는다. 핵 파동함수 χ를 보유하지 않으므로 exact α, b, ε²를 CTMQC ensemble에서 고유하게 복원하지도 않는다.')
    page();title('9. 실제 측정 결과: 성공과 실패 모두 기록')
    rows=collect(roots)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    receipt=out.with_suffix('.json');receipt.write_text(json.dumps(rows,indent=2)+'\n')
    csvpath=out.with_suffix('.csv')
    keys=['path','model','basis','K','regularization','constraint','status','last_saved_fs','initial_state_l2','max_state_l2','matched_initial_l2','max_density_l1','photon_error','population_error','max_energy_drift','wall_seconds','nfev','error']
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    para(f'이 PDF 생성 시 발견한 고정 R 실행은 {len(rows)}개다. 아래 표에는 실행된 10 fs 이상 시험을 표시한다. '
         '모든 짧은 시험, 실패/예산 종료와 full 경로는 동반 JSON/CSV에 보존한다. '
         '최대 L2는 복소 상태 norm 오차이며 population의 % 오차가 아니다. wall time은 이 로컬 CPU에서 측정한 값으로 서버 예측 시간이 아니다.')
    selected=[r for r in rows if r['last_saved_fs']>=9.99]
    data=[['실행 폴더 이름','기저/K','도달 fs','최대 L2','ΔE Ha','벽시계 s']]
    for r in selected:
        invalid=r.get('quality_audit',{}).get('first_invalid_diagnostic_fs')
        label=Path(r['path']).name+(f' [INVALID >= {invalid:g} fs]' if invalid is not None else '')
        data.append([label,f"{r['basis']}/{r['K']}",f"{r['last_saved_fs']:.1f}",f"{r['max_state_l2']:.3g}" if invalid is None else '해석 불가',f"{r['max_energy_drift']:.2g}",f"{r['wall_seconds']:.1f}"])
    table(data,[190,55,42,60,60,68])
    para('해석: K를 늘려도 초기 배치, tangent 조건수, regularization과 적분 허용오차 때문에 단조 수렴을 보장하지 않는다. '
         '에너지 constraint on/off는 같은 max-step 및 tolerance에서 비교해야 한다. BO/CBO 비교에서는 각 초기 fitting 오차를 반드시 포함한다. '
         '한 조건에서 좋은 K를 다른 결합·시간·분자에 그대로 적용할 근거는 없다.')
    para('장시간 한계: MCEF nx=17의 40 fs 탐색에서는 13 fs부터 reference/국소 근사 영역 문제가 검출되었다. '
         '그 이후의 큰 L2 값은 Gaussian 방법의 정확도 비교에 사용할 수 없다. Real-electron CBO는 nquad 32와 64의 matrix 차이가 약 6.94%였고, '
         '이제 preflight에서 미수렴으로 거부한다. 이전의 6 fs 탐색은 신뢰할 결과가 아니다.')
    for root in roots:
        for p in sorted(Path(root).rglob('accuracy_audit.json')):
            title('CTMQC 비교: '+p.parent.name)
            audit=json.loads(p.read_text());data=[['방법','공통 fs','photon MAE','S1 MAE','전체 wall s']]
            for name,r in audit.items():
                sp=p.parent/name/'summary.json';s=json.loads(sp.read_text())
                data.append([name+(' [FAILED]' if s['status']=='FAILED' else ''),f"{r['through_fs']:.1f}",
                    f"{r['photon_number']['mae']:.5g}",f"{r['bo_population']['mae']:.5g}",f"{s['wall_seconds']:.1f}"])
            table(data,[150,60,85,85,95])
            para('오차는 모든 방법이 저장된 공통 시간 구간에서 비교한다. wall time은 실패 전 시도까지 포함한 전체 실행이다. '
                 '작은 Ntraj와 periodic-QD/open-trajectory 경계 차이를 포함하므로 수렴된 물리적 우열이 아니다.')
    page();title('10. 비용과 사용 가치 판단 기준')
    para('고정 R 저장 상태의 복소 변수 수는 K(Ne+1)이다. 전역 q-grid의 Ne×Nq를 유지하지 않는다. '
         'Coefficient overlap 처리는 대략 K³, 상태별 contraction은 Ne와 K, CBO quadrature는 K²×Nquad 및 전자 diagonalization에 의존한다. '
         'K가 두 배라고 전체 cost가 정확히 두 배가 되지 않는다. adaptation의 RHS 호출 수도 달라진다.')
    para('Gaussian-POL adapter는 위 cost에 기존 polaritonic CT force 계산을 추가하므로, 현재 구현에서 POL보다 빠르다고 미리 가정할 수 없다. '
         'benchmark.py는 같은 ensemble, nf에서 native EL/POL과 Gaussian RHS를 측정한다. 전체-run 시간과 microbenchmark를 구분해야 한다. '
         '사용 목표인 “EL보다 정확하면서 POL보다 싸다”는 두 조건을 동시에 통과해야 한다. 이 조건이 확인되지 않으면 연구 prototype으로 남겨야 한다.')
    for root in roots:
        for p in sorted(Path(root).glob('cost_nf*.json')):
            cost=json.loads(p.read_text());title('측정: '+p.name)
            data=[['방법','K','Ntraj','Nf','RHS median ms']]
            for r in cost['rows']:data.append([r['method'],r.get('K','—'),r['ntraj'],r['nf'],f"{1000*r['median_seconds']:.3f}"])
            table(data,[130,55,65,65,160])
            para('초기상태에서의 RHS 측정이다. 동시 실행/OS scheduling의 영향이 있으므로 min 및 반복 수는 JSON을 참고한다. 이 표만으로 40 fs 속도비를 단정하지 않는다.')
    page();title('11. 코드와 재현 방법')
    for name,description in [('core.py','coherent overlap, complex tangent equation, Schur, energy constraint, C quotient'),('models.py','PEN analytic matrices, real-electron PEN, CBO derivative, MCEF local surrogate'),('experiment.py','fixed-R reference, K tests, actual residual, metrics/checkpoints'),('compare.py','native EL/POL/QD and experimental Gaussian-projected CT closure'),('benchmark.py','instantaneous cost and QD observable errors'),('render.py','density, current, population, error PNG/PDF/MP4'),('derive_report.py','this document and machine-readable receipts')]:
        para('<b>'+name+'</b>: '+description)
    para('모든 파일은 multi_component_exact_factorization/coherent_gaussian/ 아래에 있다. '
         '실행 명령과 한계는 README.rst에 기록했다. 새 outdir만 허용하며 기존 계산/체크포인트를 덮어쓰지 않는다. '
         'checkpoint.npz는 recovery용 상태 snapshot이며 이번 driver에는 자동 restart CLI를 제공하지 않는다. 중단 후 이어 돌렸다고 주장하지 않는다.')
    title('12. 참고문헌과 구현 범위')
    refs=[
        ('Shalashilin & Burghardt, J. Chem. Phys. 129, 084104 (2008)','https://doi.org/10.1063/1.2969101','복소 Gaussian 계수/parameter 변분 전파. 정규화된 coherent gauge로 다시 전개했다.'),
        ('Vacher, Bearpark & Robb, Theor. Chem. Acc. 135, 187 (2016)','https://doi.org/10.1007/s00214-016-1937-2','single-set vMCG의 coefficient와 Schur parameter 식. 전체 논문 코드를 복제한 것은 아니다.'),
        ('Wang et al., J. Phys. Chem. B (2021)','https://doi.org/10.1021/acs.jpcb.1c00887','multi-D2의 전자별 복소계수와 공유 coherent displacement 표현.'),
        ('Joubert-Doriol, JCTC 18 (2022)','https://doi.org/10.1021/acs.jctc.2c00461','near-linear-dependence와 rank 문제. 논문의 finite-step semiunitary algorithm 전체를 구현하지 않았다.'),
        ('Schnappinger & Kowalewski, JCTC (2023)','https://doi.org/10.1021/acs.jctc.2c01154','CBO의 photon-coordinate derivative coupling을 생략하지 않아야 하는 이유.'),
        ('Lasser & Su, arXiv:2103.11783','https://arxiv.org/abs/2103.11783','실수/복소 변분 다양체에서 서로 다른 변분 원리의 구분.')]
    for name,url,note in refs:para(escape(name)+f'<br/><link href="{url}">{url}</link><br/>'+note)
    para('사용자 제공 MCEF_Gaussian_Trajectory_Codex_Prompt.md, Reference.pdf/tex, 11_3.pdf, CTMQC and another approximation of MCEF 노트도 읽었다. '
         '기존 positive-bridge/MLS 계약은 바꾸지 않았으며, 이번 complex-G 경로는 명시적 별도 실험으로 분리했다.')
    def footer(canvas,doc):
        canvas.setFont(font,8);canvas.drawString(45,24,'MCEF | fixed-width complex Gaussian | experimental validation')
        canvas.drawRightString(A4[0]-45,24,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=42,
        title='MCEF complex Gaussian derivation and validation').build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--roots',type=Path,nargs='+',required=True)
    a=p.parse_args();report(a.out,a.roots)


if __name__=='__main__':main()
