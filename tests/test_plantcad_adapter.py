import unittest, subprocess, sys, tempfile
from pathlib import Path
from popgenlm.plantcad import coordinate_to_context, harmonized_score, raw_score, reverse_complement, validate_reference

class TestPlantCADAdapter(unittest.TestCase):
    def test_coordinate_and_padding(self):
        c,i=coordinate_to_context('ACGT',1); self.assertEqual(i,255); self.assertEqual(len(c),512); self.assertEqual(c[255],'A'); self.assertTrue(c.startswith('N'))
        c,_=coordinate_to_context('ACGT',4); self.assertEqual(c[255],'T'); self.assertTrue(c.endswith('N'))
    def test_mapping_and_ref_rejection(self):
        validate_reference('ACGT',2,'C')
        with self.assertRaises(ValueError): validate_reference('ACGT',2,'A')
        with self.assertRaises(ValueError): coordinate_to_context('ACGT',0)
    def test_orientation_and_score(self):
        self.assertEqual(reverse_complement('ACGTN'),'NACGT')
        self.assertEqual(raw_score([1.,2.,3.,4.],'A','T'),3.)
        self.assertEqual(harmonized_score(3.),-3.)

    def test_entrypoint_preflight_from_different_cwd(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path=Path(tmp)
            checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir()
            fixture=tmp_path/'fixture.tsv'
            fixture.write_text('variant_id\tpos\tref\talt\tfasta_chrom\n' + ''.join(f'v{i}\t1\tA\tC\tchr1\n' for i in range(100)))
            reference=tmp_path/'reference.fa'; reference.write_text('')
            upstream=tmp_path/'zero_shot_score.py'; upstream.write_text('')
            output=tmp_path/'output.tsv'
            r=subprocess.run([sys.executable,'-I','-B','-u',str(root/'scripts/plantcad_compare.py'),'--checkpoint',str(checkpoint),'--fixture',str(fixture),'--reference',str(reference),'--output',str(output),'--upstream-zero-shot-score',str(upstream),'--preflight-only'],cwd=tmp,text=True,capture_output=True,check=False)
        self.assertEqual(r.returncode,0,r.stderr); self.assertIn('PREFLIGHT_PASS',r.stdout)

    def test_comparison_validation(self):
        from popgenlm.plantcad import comparison_pass, validate_comparison_ids
        good=[{'variant_id':'a','adapter_batch1':0.,'authors_reference':0.,'adapter_batch8':0.,'reverse_complement':0.}]
        validate_comparison_ids(good,['a'])
        with self.assertRaises(ValueError): validate_comparison_ids([],['a'])
        with self.assertRaises(ValueError): validate_comparison_ids(good+good,['a','a'])
        bad=dict(good[0]); bad['authors_reference']=float('nan')
        with self.assertRaises(ValueError): validate_comparison_ids([bad],['a'])
        self.assertFalse(comparison_pass(1.,1.01)); self.assertTrue(comparison_pass(1.,1.00005))

if __name__ == '__main__': unittest.main()
