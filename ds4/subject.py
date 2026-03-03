from data.subject import SubjectData


class SubjectDataDs4(SubjectData):

    def label_names(self) -> tuple:
        return ("handL", "handR", "passive", "legL", "tongue", "legR")

    def channel_names(self) -> tuple:
        return (
            "Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","A1","A2",
            "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"
        )
