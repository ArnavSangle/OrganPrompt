CLASS_NAMES = {
    0: "bladder",
    1: "femur-left",
    2: "femur-right",
    3: "heart",
    4: "kidney-left",
    5: "kidney-right",
    6: "liver",
    7: "lung-left",
    8: "lung-right",
    9: "pancreas",
    10: "spleen",
}

# Zero-shot: concise anatomical description with image-space orientation.
# "image left/right/top/bottom" refers to position within the 2D axial slice.
ZERO_SHOT_TEMPLATES = {
    0:  "Axial CT scan showing the urinary bladder, a fluid-filled hollow organ appearing as a smooth hypodense oval. Located at the horizontal center of the image, in the top half, equidistant from the left and right image edges.",
    1:  "Axial CT scan showing the left femur in cross-section, appearing as a bright dense cortical ring with softer cancellous interior. Located on the right side of the image, roughly vertically centered.",
    2:  "Axial CT scan showing the right femur in cross-section, appearing as a bright dense cortical ring with cancellous medullary canal. Located on the left side of the image, roughly vertically centered.",
    3:  "Axial CT scan showing the heart, a soft tissue mass with visible chamber boundaries. Located slightly right of image center, in the top half of the image.",
    4:  "Axial CT scan showing the left kidney, a bean-shaped soft tissue organ with visible cortex and medulla. Located on the right side of the image, in the bottom half, lateral to the spine.",
    5:  "Axial CT scan showing the right kidney, a bean-shaped soft tissue organ. Located on the left side of the image, in the bottom half, lateral to the spine.",
    6:  "Axial CT scan showing the liver, a large homogeneous soft tissue mass. Occupies the left portion of the image, spanning from the left edge toward the center, positioned in the top half.",
    7:  "Axial CT scan showing the left lung, a low-density aerated structure with branching vascular markings. Occupies the right half of the image, with the mediastinum along its left border.",
    8:  "Axial CT scan showing the right lung, a low-density aerated structure with branching vascular markings. Occupies the left half of the image, with the mediastinum along its right border.",
    9:  "Axial CT scan showing the pancreas, an elongated soft tissue structure. Spans horizontally across the center of the image, with the wider head toward the left side and the tail extending toward the right side, in the bottom half of the image.",
    10: "Axial CT scan showing the spleen, a homogeneous soft tissue organ with a smooth capsule. Located on the right side of the image, in the bottom half, with its convex surface facing the right image edge.",
}

# Few-shot: richer descriptions with image-space orientation, adjacent structures,
# and imaging appearance detail.
FEW_SHOT_TEMPLATES = {
    0:  "Axial CT scan of the pelvis showing the urinary bladder as a smooth-walled fluid-filled hypodense oval. It sits at the horizontal center of the image in the top half, symmetric across the vertical midline. The rectum appears as a smaller structure at the bottom of the image behind it. The pubic symphysis is visible at the very top image edge in front of it. Wall thickness is thin and uniform.",
    1:  "Axial CT scan of the thigh showing the left femur in transverse cross-section. Appears on the RIGHT side of the image as a bright hyperdense cortical ring surrounding lower-density cancellous bone with a central medullary canal. Adductor muscles border it toward the image center (medially). Vastus lateralis and iliotibial band flank it toward the right image edge (laterally).",
    2:  "Axial CT scan of the thigh showing the right femur in transverse cross-section. Appears on the LEFT side of the image as a hyperdense cortical ring with cancellous medullary interior. Adductor muscles border it toward the image center (medially). Vastus lateralis flanks it toward the left image edge (laterally). Slightly smaller cross-sectional area than the left femur in most individuals.",
    3:  "Axial CT scan of the chest showing the heart within the pericardial sac. Appears slightly right of image center in the top half. The cardiac apex points toward the bottom-right of the image. The right ventricle is nearest the top of the image (most anterior), the left ventricle toward the bottom-right. Both lungs flank it — right lung on the image left, left lung on the image right. Pericardial fat appears as low-density material surrounding the cardiac silhouette.",
    4:  "Axial CT scan of the upper abdomen showing the left kidney in the retroperitoneal space. Appears on the RIGHT side of the image in the bottom half. Its long axis runs roughly vertically. The renal hilum faces toward the image center (medially). The aorta is visible toward the bottom-center of the image just to its left. Perirenal fat surrounds the organ. Renal cortex is slightly brighter than the medullary pyramids.",
    5:  "Axial CT scan of the upper abdomen showing the right kidney in the retroperitoneal space. Appears on the LEFT side of the image in the bottom half, slightly more toward the top than the left kidney. The renal hilum faces toward the image center (medially). The inferior vena cava is visible just to its right in the image. The liver occupies the top-left of the image directly above it.",
    6:  "Axial CT scan of the upper abdomen showing the liver. Dominates the LEFT portion of the image from the left edge toward and past the center, in the top half. The right lobe is largest and sits furthest left in the image. The left lobe tapers toward image center. Intrahepatic vessels (portal and hepatic veins) branch throughout as tubular structures. The gallbladder may appear as a small oval hypodense structure at the bottom of the liver mass.",
    7:  "Axial CT scan of the chest showing the left lung. Occupies the RIGHT half of the image as low-attenuation aerated tissue. The mediastinum and heart border it along the left side of the right half. The chest wall curves along the right image edge. Branching bronchial and vascular markings extend from the hilum toward the periphery. The left lung is slightly smaller than the right due to the heart pushing into the left hemithorax.",
    8:  "Axial CT scan of the chest showing the right lung. Occupies the LEFT half of the image as low-attenuation aerated tissue. The mediastinum borders it along the right side of the left half. The chest wall curves along the left image edge. Three lobes are present, divided by fissures visible as thin lines. Branching bronchial and vascular markings radiate from the hilum. The right lung is larger than the left.",
    9:  "Axial CT scan of the upper abdomen showing the pancreas in the retroperitoneal space. Lies horizontally across the bottom half of the image. The head (widest portion) is on the LEFT side of the image, nestled in the duodenal loop. The body crosses the center of the image. The tail extends toward the RIGHT side of the image, pointing toward the spleen. The superior mesenteric vessels appear as round structures just in front of the pancreatic body.",
    10: "Axial CT scan of the upper left abdomen showing the spleen. Appears on the RIGHT side of the image in the bottom half. Its convex surface faces the right image edge and the lateral chest wall. Its concave hilum faces toward the image center-left. The stomach may be visible just to its left in the image. The left kidney is visible just below and behind it. Homogeneous soft tissue attenuation with a smooth capsule.",
}
