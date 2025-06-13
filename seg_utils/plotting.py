import matplotlib.pyplot as plt

#important vertex
RL = [0, 21, 29]
LL = [0, 21, 27, 40, 44]
H = [0, 6, 12, 18]

def reverseVector(vector):
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    
    p1 = RLUNG*2
    p2 = p1 + LLUNG*2
    p3 = p2 + HEART*2
    
    rl = vector[:p1].reshape(-1,2)
    ll = vector[p1:p2].reshape(-1,2)
    h = vector[p2:p3].reshape(-1,2)

    return rl, ll, h

def draw_organ(ax, array, color = 'b', bigger = None):
    N = array.shape[0]
    
    for i in range(0, N):
        x, y = array[i,:]
        
        if bigger is not None:
            if i in bigger:
                circ = plt.Circle((x, y), radius=9, color=color, fill = True)
                ax.add_patch(circ)
                circ = plt.Circle((x, y), radius=3, color='white', fill = True)
            else:
                circ = plt.Circle((x, y), radius=3, color=color, fill = True)
        else:
            circ = plt.Circle((x, y), radius=3, color=color, fill = True)
            
        ax.add_patch(circ)
    return

def draw_lines(ax, array, color = 'b'):
    N = array.shape[0]
    
    for i in range(0, N):
        x1, y1 = array[i-1,:]
        x2, y2 = array[i,:]
        
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)
    return

def draw_single_organ(ax, organ_vector, color, organ_shape=None):
    draw_lines(ax, organ_vector, color)
    if organ_shape is not None:
        draw_organ(ax, organ_vector, color, organ_shape)

def show_each_organ(vector, img=None):
    vector = vector.reshape(-1, 1)
    right_lung, left_lung, heart = reverseVector(vector)
    organs = [
        ("Right Lung", right_lung, 'r', RL),
        ("Left Lung", left_lung, 'g', LL),
        ("Heart", heart, 'y', H),
    ]

    for title, organ_vec, color, shape in organs:
        fig, ax = plt.subplots()
        if img is not None:
            ax.imshow(img, cmap='gray')
        draw_single_organ(ax, organ_vec* 1024, color, shape)
        ax.set_title(title)
        plt.show()
